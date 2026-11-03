from types import SimpleNamespace
import pytest
from echo_routing.temporal.structured_guard import storage_bytes

def test_removed_atomic_pending_file_does_not_abandon_a_live_guard():
    runner=lambda *a,**k:SimpleNamespace(returncode=1,stdout='1024 /a\n2048 /b\n',stderr="du: cannot access '/a/pending.json': No such file or directory\n")
    assert storage_bytes(['/a','/b'],runner)==105003072
    with pytest.raises(RuntimeError):storage_bytes(['/a','/b'],lambda *a,**k:SimpleNamespace(returncode=1,stdout='1024 /a\n2048 /b\n',stderr='permission denied'))
    with pytest.raises(RuntimeError):storage_bytes(['/a','/b'],lambda *a,**k:SimpleNamespace(returncode=0,stdout='1024 /a\n',stderr=''))
