In de trqa folder zitten en zet env aan:
```source .venv/bin/activate```

Download de modellen:
```export HF_HOME=/vol/csedu-nobackup/course/I00041_informationretrieval/users/$USER/hf_cache```

```hf download Qwen/Qwen2.5-7B-Instruct```

```hf download PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.3```

en installeer deze package:
```uv pip install accelerate```


Dan verder de scripts van de c5 folder zitten al in de trqa code maar moeten aangepast worden. En de generation.sh moet in de root van die trqa folder zelf. en check de generation.sh files, die kun je configureren zoals bij de METHOD variable welk model je wilt gebruiken.
