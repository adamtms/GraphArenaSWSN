
- Problems: We cannot use the other external libraries, since they do not allow for token manipulation
- Currently embedding size is dependent in 100% on embedding size of llm, maybe using more tokens, e.g. 2 consecutive tokens would make it better
- LION optimizer - should be better than what we use (just add it to the pyproject.toml)
- Constrained decoding - task already in github
- Bigger GNNs, Better Model, more epochs (most were stopped due to finishing iteration instead of the loss staying the same)
- Maybe try the normal problems (that return a list) instead of just a number
- Fix prompt for Connected (it should return 1 or 2)
