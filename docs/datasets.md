# Datasets

## LFW (Labeled Faces in the Wild)

- 13,233 images, 5,749 identities
- Pre-aligned (funneled version)
- Source: http://vis-www.cs.umass.edu/lfw/
- Mirror: https://ndownloader.figshare.com/files/5976015
- Extraction script: `scripts/extract_lfw_embeddings.py`

## VGGFace2

- 3.31M images, 9,131 identities (full dataset)
- We use test split streamed from HuggingFace: `logasja/VGGFace2`
- 100K embeddings extracted (291 identities from streaming order)
- Extraction script: `scripts/extract_vggface2_fast.py`

## Not Yet Tested

| Dataset | Status | Reason |
|---|---|---|
| IJB-C | Requires academic access agreement | Apply at https://nigos.nist.gov/ |
| MegaFace | 159GB download | Requires compute time |
| MS-Celeb-1M | Restricted/withdrawn | Privacy concerns |
