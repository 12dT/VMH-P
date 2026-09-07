# D2R Backbone Notes

D2R is consumed through `adapters/d2r_adapter.py`. The main TreeRing
interface reads only existing D2R tensors:

- `text`: pre-interaction BERT global/CLS evidence
- `vision`: pre-interaction ViT global/CLS evidence
- `multimodal`: final Block-Fusion feature before D2R's classifier
- `base_logits`: D2R's original classifier logits

D2R router probabilities and routed branch features are optional analysis
signals and are not required inputs for the frozen plug-in method.
