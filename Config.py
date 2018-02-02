BatchSize = 50

VocabSize = 1000
EmbeddingSize = 512
EncoderPreNetConvKernelSize = 5
EncoderPreNetConvFilterSize = 512
EncoderPreNetStackSize = 3
EncoderHiddenSize = 256
MaxSourceLength = 128
EncoderDropoutRate = 0.5

AttentionPositionHiddenSize = 32
AttentionConvFilterSize = 32
AttentionConvKernelSize = 31

DecoderDropoutRate = 0.5
DecoderPreNetHiddenSize = 256
DecoderPreNetStackSize = 2
DecoderRnnStackSize = 2
DecoderHiddenSize = 1024
DecoderPostNetHiddenSize = 512
DecoderPostNetConvKernelSize = 5
DecoderPostNetConvFilterSize = 512
DecoderPostNetStackSize = 5
AttentionSize = 128
MaxTargetLength = 128
NumMels = 80

FrameShiftMs = 12.5

# Comma-separated list of cleaners to run on text prior to training and eval. For non-English
# text, you may want to use "basic_cleaners" or "transliteration_cleaners" See TRAINING_DATA.md.
Cleaners = "english_cleaners"

LogDir = "./log"
DataDir = "./training/train.txt"
GroupSize = 8
CheckpointInterval = 500

BufferSize = 16

MelVectorSize = 80

GPUIndex = '3'



