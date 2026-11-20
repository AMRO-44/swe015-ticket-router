# What you still have to do (≈ 2 hours)  –  delete this file before submitting

1. **GitHub** – create a public repo `swe015-ticket-router`, push this folder
   (`git init && git add . && git commit -m "SWE015 homework" && git push`). `models/` is ignored on purpose.
2. **Hugging Face** – create a free account + a *write* token (https://huggingface.co/settings/tokens).
3. **Edit 3 placeholders**
   * `src/config.py` → `HF_REPO_ID = "<your-hf-username>/swe015-ticket-router"`
   * `notebooks/colab_train.ipynb` → your GitHub URL and HF repo id
   * `slides_info.json` → name, student ID, repo URL, video URL, HF URL
   Commit + push.
4. **Train on Colab** – open the notebook in Colab, *Runtime → T4 GPU*, run all cells
   (train ≈ 30–45 min, upload weights, download `outputs.zip`).
5. **Commit the results** – unzip `outputs.zip` into the repo (`results/`, `docs/`) and push, so the
   repo, the slides and the README all show YOUR numbers.
6. **Check the slides** – open `docs/presentation.pptx`, read every slide (the text on slides 10–11 is
   generated from your numbers – adjust wording if needed), then export as **PDF**.
7. **Test the README in a fresh environment** – e.g. a new Colab: clone, `pip install -r requirements.txt`,
   `python app.py --share`; it must download the weights from the Hub and start.
8. **Record the 2-minute video** (narrated, upload unlisted, test the link in a private window):
   * 0:00–0:15 problem in one sentence
   * 0:15–1:15 live demo: type `my dishwaser is leaking water everywhere` (spelling correction),
     `bulaşık makinem su boşaltmıyor, model DW-6534` (Turkish + model code),
     `what is the best pizza topping on mars` (difficult: out-of-scope → human agent), `hello!` (ELIZA)
   * 1:15–1:45 architecture slide + results table
   * 1:45–2:00 conclusion
9. Put the video link into `slides_info.json`, regenerate (`python scripts/make_slides.py`) or edit
   slide 1 by hand, export PDF, submit PDF + video link + repo link.

Be ready to explain: backprop in `src/ffn_np.py`, SGNS in `src/word2vec_np.py`, edit distance,
distillation loss and dynamic INT8 quantization in `src/transformer.py`.
