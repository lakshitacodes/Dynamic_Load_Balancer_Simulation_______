"""
dynamic_load_balancer.py
A simulation of an adaptive dynamic load balancer with:
- least-loaded assignment
- work-stealing by idle workers
- auto-scaling of worker count (up/down)
Run with: python3 dynamic_load_balancer.py
"""

import threading
import queue
import time
import random
import statistics
from typing import Optional

# -------------------------------
# Task definition
# -------------------------------
class Task:
    def __init__(self, task_id: int, cost: float):
        """
        cost: estimated processing time in seconds (simulated)
        """
        self.task_id = task_id
        self.cost = cost  # seconds

    def run(self):
        # Simulate CPU work by sleeping for 'cost' seconds.
        # In a real system this would be CPU-bound work.
        time.sleep(self.cost)

    def __repr__(self):
        return f"Task(id={self.task_id}, cost={self.cost:.2f})"


# -------------------------------
# Worker definition
# -------------------------------
class Worker(threading.Thread):
    def __init__(self, worker_id: int, scheduler, retire_event: threading.Event):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.scheduler = scheduler
        self.queue = queue.Queue()
        self.current_task: Optional[Task] = None
        self.lock = threading.Lock()
        self.total_processed = 0
        # Event that signals the scheduler requests this worker to retire
        self.retire_event = retire_event
        self.last_active = time.time()
        self.running = True

    def enqueue(self, task: Task):
        self.queue.put(task)

    def get_estimated_load(self) -> float:
        # Estimated load = sum of costs of queued tasks + remaining of current task (approx)
        qsize = 0.0
        if not self.queue.empty():
            # sum of costs in queue: read without removing by iterating queue.queue (safe-ish for simulation)
            try:
                qlist = list(self.queue.queue)
                qsize = sum(t.cost for t in qlist)
            except Exception:
                # fallback to count only
                qsize = self.queue.qsize()
        current = self.current_task.cost if self.current_task else 0.0
        return current + qsize

    def try_steal(self) -> bool:
        # Ask scheduler for a task to steal
        stolen = self.scheduler.steal_task(from_worker=self.worker_id)
        if stolen:
            self.enqueue(stolen)
            return True
        return False

    def run(self):
        while self.running:
            try:
                task: Task = self.queue.get(timeout=0.5)
            except queue.Empty:
                # no task — try to steal if scheduler supports it
                if self.retire_event.is_set():
                    # scheduler asked this worker to retire if it's idle
                    # only retire if still idle
                    if self.queue.empty() and self.current_task is None:
                        self.running = False
                        self.scheduler.log(f"Worker-{self.worker_id} retiring (idle).")
                        break
                # attempt a steal
                self.try_steal()
                continue

            with self.lock:
                self.current_task = task
            self.last_active = time.time()
            self.scheduler.log(f"Worker-{self.worker_id} START {task}")
            # Run task (simulated)
            task.run()
            with self.lock:
                self.current_task = None
                self.total_processed += 1
            self.scheduler.log(f"Worker-{self.worker_id} DONE {task}")
            self.queue.task_done()
        self.scheduler.log(f"Worker-{self.worker_id} STOPPED.")


# -------------------------------
# Scheduler definition
# -------------------------------
class Scheduler:
    def __init__(
        self,
        min_workers=2,
        max_workers=10,
        scale_up_threshold=4.0,   # avg backlog (seconds) per worker above which we scale up
        scale_down_threshold=1.0, # avg backlog (seconds) per worker below which we scale down
        monitor_interval=2.0,
    ):
        self.workers = {}  # id -> Worker
        self.worker_id_seq = 0
        self.workers_lock = threading.Lock()
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.monitor_interval = monitor_interval
        self.stop_event = threading.Event()
        self.retire_events = {}  # worker_id -> Event used to signal retirement
        self.log_lock = threading.Lock()

        # Start with min_workers
        for _ in range(self.min_workers):
            self.add_worker()

        # background monitor thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def log(self, msg: str):
        with self.log_lock:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] {msg}")

    def add_worker(self):
        with self.workers_lock:
            wid = self.worker_id_seq
            retire_event = threading.Event()
            w = Worker(wid, scheduler=self, retire_event=retire_event)
            self.workers[wid] = w
            self.retire_events[wid] = retire_event
            self.worker_id_seq += 1
            w.start()
            self.log(f"Worker-{wid} ADDED. TotalWorkers={len(self.workers)}")
            return wid

    def remove_worker(self, wid: int):
        # Mark worker to retire when idle; we don't force-kill a running task here.
        with self.workers_lock:
            if wid not in self.workers:
                return False
            self.retire_events[wid].set()
            # actual removal happens when worker thread exits
            self.log(f"Worker-{wid} scheduled for retirement.")
            return True

    def _current_workers_list(self):
        with self.workers_lock:
            return list(self.workers.items())

    def _monitor_loop(self):
        while not self.stop_event.is_set():
            time.sleep(self.monitor_interval)
            self._monitor_and_scale()

    def _monitor_and_scale(self):
        workers_snapshot = self._current_workers_list()
        if not workers_snapshot:
            return
        loads = []
        qlens = []
        now = time.time()

        for wid, w in workers_snapshot:
            try:
                est = w.get_estimated_load()
            except Exception:
                est = 0.0
            loads.append(est)
            qlens.append(w.queue.qsize())

        avg_load = statistics.mean(loads) if loads else 0.0
        max_load = max(loads) if loads else 0.0
        total_workers = len(workers_snapshot)

        self.log(
            f"MONITOR: Workers={total_workers} AvgLoadSec={avg_load:.2f} MaxLoadSec={max_load:.2f} Qlens={qlens}"
        )

        # Scale up
        if avg_load > self.scale_up_threshold and total_workers < self.max_workers:
            self.add_worker()

        # Scale down: retire an idle worker if average load is low
        if avg_load < self.scale_down_threshold and total_workers > self.min_workers:
            # prefer to retire the worker with smallest activity and no queued tasks
            candidate = None
            idle_threshold = 2.0  # seconds of inactivity
            for wid, w in reversed(workers_snapshot):  # reversed to prefer recently added
                if w.queue.empty() and (now - w.last_active) > idle_threshold and w.current_task is None:
                    candidate = wid
                    break
            if candidate is None:
                # fallback: pick the worker with smallest estimated load
                min_wid = min(workers_snapshot, key=lambda it: it[1].get_estimated_load())[0]
                candidate = min_wid
            if candidate is not None:
                self.remove_worker(candidate)

        # Clean up stopped workers from dict
        with self.workers_lock:
            to_delete = []
            for wid, w in list(self.workers.items()):
                if not w.is_alive():
                    to_delete.append(wid)
            for wid in to_delete:
                del self.workers[wid]
                del self.retire_events[wid]
                self.log(f"Worker-{wid} removed from registry. TotalWorkers={len(self.workers)}")

    def submit_task(self, task: Task):
        # Choose least-loaded worker
        with self.workers_lock:
            if not self.workers:
                # no workers: create one
                self.add_worker()
            # compute estimated loads
            best_wid = None
            best_load = float("inf")
            for wid, w in self.workers.items():
                try:
                    est = w.get_estimated_load()
                except Exception:
                    est = 0.0
                if est < best_load:
                    best_load = est
                    best_wid = wid
            # enqueue task
            if best_wid is not None:
                self.workers[best_wid].enqueue(task)
                self.log(f"DISPATCH: {task} -> Worker-{best_wid} (est_load={best_load:.2f})")
            else:
                # fallback to random
                wid = random.choice(list(self.workers.keys()))
                self.workers[wid].enqueue(task)
                self.log(f"DISPATCH-FALLBACK: {task} -> Worker-{wid}")

    def steal_task(self, from_worker: int) -> Optional[Task]:
        """
        Called by an idle worker to steal a single task from the busiest worker.
        Returns a Task or None.
        """
        with self.workers_lock:
            # find busiest worker (exclude the thief itself)
            candidates = [(wid, w) for wid, w in self.workers.items() if wid != from_worker]
            if not candidates:
                return None
            busiest_wid, busiest = max(candidates, key=lambda it: it[1].get_estimated_load())
            # attempt to steal one task from busiest worker's queue
            try:
                # Non-blocking, get one item from the other worker's queue
                stolen_task = None
                # We need to synchronize on the other worker's queue to be safer
                # Here we use queue.Queue internal .mutex for safe access to its deque (acceptable in simulation)
                with busiest.queue.mutex:
                    if busiest.queue.queue:
                        stolen_task = busiest.queue.queue.pop()  # steal the last item (LIFO steal)
                if stolen_task:
                    self.log(f"STEAL: Worker-{from_worker} stole {stolen_task} from Worker-{busiest_wid}")
                    return stolen_task
            except Exception:
                return None
        return None

    def shutdown(self, wait=True):
        self.stop_event.set()
        self.log("Scheduler shutting down: signaling all workers to retire.")
        with self.workers_lock:
            for wid, ev in self.retire_events.items():
                ev.set()
        if wait:
            # join all workers
            while True:
                with self.workers_lock:
                    if not self.workers:
                        break
                    current_workers = list(self.workers.values())
                for w in current_workers:
                    w.join(timeout=0.1)
            self.monitor_thread.join(timeout=0.5)
            self.log("Scheduler shutdown complete.")


# -------------------------------
# Simulation harness
# -------------------------------
def simulate(
    runtime_sec=30,
    arrival_rate=1.5,  # average tasks per second (Poisson)
    min_cost=0.1,
    max_cost=2.0,
):
    scheduler = Scheduler(min_workers=2, max_workers=8, scale_up_threshold=3.0, scale_down_threshold=0.8)
    task_id = 0
    start_time = time.time()
    next_arrival = start_time + random.expovariate(arrival_rate)
    try:
        while time.time() - start_time < runtime_sec:
            now = time.time()
            if now >= next_arrival:
                # generate a new task
                cost = random.uniform(min_cost, max_cost)
                t = Task(task_id=task_id, cost=cost)
                scheduler.submit_task(t)
                task_id += 1
                # compute next arrival
                next_arrival = now + random.expovariate(arrival_rate)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        # allow queues to drain for a bit then shutdown
        scheduler.log("Simulation complete: waiting a few seconds to drain queues...")
        time.sleep(3.0)
        scheduler.shutdown(wait=True)


if __name__ == "__main__":
    # Quick demo:
    # - runtime seconds: how long to produce arrivals
    # - arrival_rate: average number of tasks per second
    simulate(runtime_sec=40, arrival_rate=1.2, min_cost=0.2, max_cost=1.5)