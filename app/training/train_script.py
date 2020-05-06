from __future__ import absolute_import

from models import model_trainer

frozen_epochs = 30
unfrozen_encoder_epochs = 10
unfrozen_embedder_epochs = 10
total_epochs = (frozen_epochs + unfrozen_embedder_epochs +
                unfrozen_encoder_epochs)

mt = model_trainer.ModelTrainer(expected_total_epochs=total_epochs)
mt.train(frozen_epochs)
mt.unfreeze_encoder()
mt.train(unfrozen_encoder_epochs)
mt.unfreeze_embeddings()
mt.train(unfrozen_embedder_epochs)
