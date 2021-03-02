from src.csi.model_test import ChannelED, CSIData
from src.csi.models import DummyModel, SVDModel, PolyReduction
# initialize a model put model into src.csi.models
# if you are using ED scheme, please provide a `encoding` `decoding` methods
model = PolyReduction(deg=20)
# initialize channel
data = CSIData(speed=40)
channel_encoding_decoding = ChannelED(model=model, csi_repository=data,
                                      encoding_frequency=1, loss='frobenius', verbose=True)
# run testing
channel_encoding_decoding.test()

