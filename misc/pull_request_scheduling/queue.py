import random
import json

"""
Simple scheduler for pull/merge request review.

The reviewers are split into pairs, then the pairs are randomly shuffled
(with seed 'Wolf'), and the resulting list is treated as a circular queue.

In case the next reviewer pair in the queue contains the pull request author,
the next immediate pair that does not contain the author is searched for in
the queue, assigned, and pushed to the tail.

To add a merge request to the scheduler, to add it manually to the json file:
    * Copy and paste an existing request in the file
    * Change the 'id', 'title', and 'author' fields if necessary
    * Set the 'reviewers' field to null (translation of Python's None in js)
    * Run queue.py
"""


class PullRequestList():
    """
    Format of JSON file:
    {
      "cur_queue": [ [Reviewer1, Reviewer2] ]
      "pull_requests": [
                         {
                           "id": Merge request number,
                           "title": Merge request title,
                           "author": Merge request author,
                           "reviewers": [ Reviewer1, Reviewer2 ] or [] or null
                         }
                       ]
    }
    """
    def __init__(self, json_file='prs.json'):
        self.json_file = json_file
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        self.queue = data['cur_queue']
        self.pull_req_list = data['pull_requests']

    def process(self):
        for request in self.pull_req_list:
            self.assign_reviewer_pair(request)
        self.pull_req_list.sort(key=lambda request: request['id'])

    def assign_reviewer_pair(self, pull_request):
        if not pull_request['reviewers']:
            index = 0
            reviewer_pair = self.queue[index]
            while pull_request['author'] in reviewer_pair:
                index += 1
                reviewer_pair = self.queue[index]
            pull_request['reviewers'] = reviewer_pair
            self.queue.pop(index)
            self.queue.append(reviewer_pair)

    def print(self):
        print("==== UPDATED QUEUE ====")
        queue_strings = ( f'({first}, {second})' for first, second in self.queue )
        print('\n'.join(queue_strings))
        print("==== PULL REQUEST REVIEWER LIST ====")
        for request in self.pull_req_list[::-1]:
            request_strings = ( f'{key}={value}' for key, value in request.items() )
            print('\n'.join(request_strings))
            print('------------------------')

    def save(self):
        data = dict()
        data['cur_queue'] = self.queue
        data['pull_requests'] = self.pull_req_list[::-1]
        with open(self.json_file, 'w') as f:
            json.dump(data, f, indent=2)


def init_queue():
    reviewers = ('Ilia',
                 'Parth',
                 'Xiaoyu',
                 'Zhicheng',)
    num_reviewers = len(reviewers)

    reviewer_queue = []
    for i in range(num_reviewers):
        for j in range(i+1, num_reviewers):
            reviewer_queue.append([reviewers[i], reviewers[j]])

    seed = sum( ord(char) for char in 'Wolf' )
    random.seed(seed)
    random.shuffle(reviewer_queue)
    return reviewer_queue


if __name__ == "__main__":
    pull_req_list = PullRequestList()
    pull_req_list.process()
    pull_req_list.print()
    pull_req_list.save()
