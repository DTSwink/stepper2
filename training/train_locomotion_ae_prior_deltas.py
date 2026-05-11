from __future__ import annotations

import train_locomotion_ae_prior as base
import transition_autoencoder_deltas as tae_deltas


base.tae = tae_deltas


if __name__ == "__main__":
    base.main()
