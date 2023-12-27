# Finetuned Google-MT5-Large

## Overview

Here is where we finetune our mt5 model, we trained 10 different models.(2 + 2 + 2*3)

```
c2e is chinese to emoji, e2c is emoji to chinese
                algo    gpt     mix
batch-2   c2e    X       X       O        
batch-2   e2c    X       X       O    
batch-4   c2e    O       O       O    
batch-4   e2c    O       O       O    
batch-8   c2e    X       X       O        
batch-8   e2c    X       X       O    
```

And we inference these ten models on test.json in ../training_data with three different ways for each two models, chinese-to-emoji(c2e), algo-emoji-to-chinese(e2c), gpt-emoji-to-chinese(e2c). We then put the result in ./predictions.