### Base class for jobs to be uploaded to the master server

class Job:
    def __iter__(self, name):
        self.name = name
        return self
    
    def __next__(self):
        if self.has_next():
            return self.next_task()
        else:
            raise StopIteration

    def reset(self):
        pass

    def has_next(self):
        raise Exception("Unimplemented")

    def total_tasks(self):
        raise Exception("Unimplemented")

    # Returns the args needed to execute the next task
    def next_task(Self):
        raise Exception("Unimplemented")
    
    # How to execute each task
    def do_task(self, args):
        raise Exception("Unimplemented")

    # Excecuted after each task, used for things like saving results to a DB
    def post_task(self):
        raise Exception("Unimplemented")
