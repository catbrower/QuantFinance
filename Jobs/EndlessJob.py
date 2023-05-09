from Jobs.Job import Job

class EndlessJob(Job):
    def total_tasks(self):
        return float('inf')
    
    def has_next(self):
        return True