import threading
from concurrent.futures import CancelledError

import pytest

from mlx_one.engine.device_executor import DeviceExecutor
from mlx_one.engine.execution_context import QueueFullError


def test_device_owns_evaluation_and_rotates_models_with_bounded_queue():
    executor=DeviceExecutor(queue_size=3)
    entered,release=threading.Event(),threading.Event()
    order=[]
    def block():
        entered.set()
        assert release.wait(2)
    first=executor.submit('load',block)
    assert entered.wait(2)
    pending=[executor.submit(k,lambda n=n: order.append(n),
             evaluate=lambda value: order.append('eval')) for k,n in [('a',1),('a',2),('b',3)]]
    with pytest.raises(QueueFullError):
        executor.submit('c',lambda:None)
    release.set()
    for future in [first,*pending]:
        future.result(2)
    assert order==[1,'eval',3,'eval',2,'eval']
    assert executor.call('a',lambda:executor.is_owner)
    assert executor.call('a',lambda:executor.call('b',lambda:7))==7
    executor.close()
    assert not executor.stats()['worker_alive']


def test_cancelled_queued_work_never_executes_and_errors_do_not_kill_owner():
    executor=DeviceExecutor()
    cancel=threading.Event()
    cancel.set()
    future=executor.submit('a',lambda:pytest.fail('cancelled work executed'),cancel=cancel)
    with pytest.raises(CancelledError):
        future.result(2)
    def fail():raise ValueError('evaluation failure')
    with pytest.raises(ValueError):
        executor.call('a',fail)
    assert executor.call('a',lambda:3)==3
    executor.close()
