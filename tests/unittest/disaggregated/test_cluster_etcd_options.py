# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""etcd credentials and TLS options of the disagg cluster configuration and storage."""

import secrets
from unittest import mock

import grpc
import pytest

from tensorrt_llm.llmapi.disagg_utils import DisaggClusterConfig, extract_disagg_cluster_config
from tensorrt_llm.serve import cluster_storage
from tensorrt_llm.serve.cluster_storage import Etcd3ClusterStorage

pytestmark = pytest.mark.cpu_only

ETCD_USER = secrets.token_hex(4)
ETCD_SECRET = secrets.token_hex(16)
RECORD_KEY = secrets.token_hex(16)
ETCD_URI = "etcd://etcd-host:2379"
OPTIONS = {
    "user": ETCD_USER,
    "password": ETCD_SECRET,
    "ca_cert": "/ca.pem",
    "cert_key": "/key.pem",
    "cert_cert": "/cert.pem",
    "record_auth_key": RECORD_KEY,
}


class _RpcError(grpc.RpcError):
    def __init__(self, code):
        super().__init__()
        self._code = code

    def code(self):
        return self._code


def test_cluster_config_etcd_options():
    assert DisaggClusterConfig(ETCD_URI).etcd_client_options() == {}
    cfg = DisaggClusterConfig(ETCD_URI, **OPTIONS)
    assert cfg.etcd_client_options() == OPTIONS
    assert ETCD_SECRET not in repr(cfg) and RECORD_KEY not in repr(cfg)


@pytest.mark.parametrize(
    "options,message",
    [
        ({"user": ETCD_USER}, "must be set together"),
        ({"user": ETCD_USER, "password": ETCD_SECRET}, "require ca_cert"),
        ({"record_auth_key": ""}, "record_auth_key must be a non-empty string"),
    ],
    ids=["user_only", "user_password_without_ca_cert", "empty_record_auth_key"],
)
def test_cluster_config_rejects_invalid_etcd_options(options, message):
    with pytest.raises(ValueError, match=message) as excinfo:
        DisaggClusterConfig(ETCD_URI, **options)
    assert ETCD_SECRET not in str(excinfo.value)


@pytest.mark.parametrize("options", [{"ca_cert": "/ca.pem"}, {"record_auth_key": RECORD_KEY}])
def test_cluster_config_etcd_options_require_an_etcd_uri(options):
    with pytest.raises(ValueError, match="require an etcd:// cluster_uri"):
        DisaggClusterConfig("http://host:8000", **options)


def test_extract_cluster_config_reads_the_options():
    # A worker takes the uri from --disagg_cluster_uri and the credentials from its config file.
    cfg = extract_disagg_cluster_config({"cluster_name": "c", **OPTIONS}, cluster_uri=ETCD_URI)
    assert cfg.cluster_uri == ETCD_URI and cfg.cluster_name == "c"
    assert cfg.etcd_client_options() == OPTIONS


def test_extract_cluster_config_validates_the_options_it_sets():
    with pytest.raises(ValueError, match="require ca_cert"):
        extract_disagg_cluster_config(
            {"cluster_uri": ETCD_URI, "user": ETCD_USER, "password": ETCD_SECRET}
        )


def test_etcd_storage_without_options_connects_as_before():
    with (
        mock.patch.object(cluster_storage.etcd3, "Endpoint") as endpoint,
        mock.patch.object(cluster_storage.etcd3, "MultiEndpointEtcd3Client") as client,
    ):
        Etcd3ClusterStorage("etcd://h1:2379,etcd://h2:2380", "c")
    assert endpoint.call_args_list == [
        mock.call("h1", 2379, secure=False, creds=None),
        mock.call("h2", 2380, secure=False, creds=None),
    ]
    client.assert_called_once_with(
        [endpoint.return_value, endpoint.return_value], timeout=10, user=None, password=None
    )


def test_etcd_storage_passes_credentials_and_tls_files_and_keeps_them_for_reconnect():
    with (
        mock.patch.object(cluster_storage.etcd3.Etcd3Client, "get_secure_creds") as secure_creds,
        mock.patch.object(cluster_storage.etcd3, "Endpoint") as endpoint,
        mock.patch.object(cluster_storage.etcd3, "MultiEndpointEtcd3Client") as client,
    ):
        storage = Etcd3ClusterStorage(ETCD_URI, "c", **OPTIONS)
        storage.reconnect()
    secure_creds.assert_called_with("/ca.pem", "/key.pem", "/cert.pem")
    endpoint.assert_called_with("etcd-host", 2379, secure=True, creds=secure_creds.return_value)
    assert client.call_count == 2
    client.assert_called_with(
        [endpoint.return_value], timeout=10, user=ETCD_USER, password=ETCD_SECRET
    )


@pytest.mark.asyncio
async def test_rejected_token_connects_again():
    stale, fresh = mock.MagicMock(), mock.MagicMock()
    stale.get.side_effect = _RpcError(grpc.StatusCode.UNAUTHENTICATED)
    fresh.get.return_value = (b"value", None)
    with mock.patch.object(Etcd3ClusterStorage, "_connect", side_effect=[stale, fresh]) as connect:
        storage = Etcd3ClusterStorage(ETCD_URI, "c", user=ETCD_USER, password=ETCD_SECRET)
        assert await storage.get("k") is None
        assert await storage.get("k") == "value"
    assert connect.call_count == 2


@pytest.mark.asyncio
async def test_other_errors_do_not_connect_again():
    client = mock.MagicMock()
    client.get.side_effect = _RpcError(grpc.StatusCode.PERMISSION_DENIED)
    with mock.patch.object(Etcd3ClusterStorage, "_connect", return_value=client) as connect:
        storage = Etcd3ClusterStorage(ETCD_URI, "c")
        with pytest.raises(grpc.RpcError):
            await storage.get("k")
    connect.assert_called_once()
