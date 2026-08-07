"""Batch inference: annotating a whole dataset with a hand-picked orchestration of models.

Three modules, in the order a run passes through them:

* :mod:`~app.services.inference.planning` -- turn a request into a frozen, hierarchy-ordered
  plan and the work list that realises it.
* :mod:`~app.services.inference.execution` -- run one unit of that work list: call the AI
  service for one image, filter the output down to the step's label, nest it under the right
  parent, drop duplicates, write it.
* :mod:`~app.services.inference.tasks` -- the Celery tasks that walk the work list.
"""
