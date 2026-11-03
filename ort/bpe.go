package ort

import (
	"unsafe"

	"github.com/oxzjh/clip"
	"github.com/oxzjh/ort"
	"github.com/oxzjh/tokenizer"
)

type Bpe struct {
	tk         *tokenizer.BpeTokenizer
	proj       map[int]map[int]float32
	engine     *ort.Ort
	inputDims  int
	outputDims int
}

func (b *Bpe) Extract(text string) ([]float32, error) {
	tokens := b.tk.Encode(text, true)
	result, err := b.engine.Infer(unsafe.Pointer(&tokens.GetInt32(b.inputDims, 0)[0]), b.inputDims<<2, ort.TypeInt32)
	if err != nil {
		return nil, err
	}
	defer result.Free()
	maxIndex := clip.GetMaxIndex(tokens)
	extracted, err := result.GetFloat32WithOffset(0, maxIndex*b.outputDims<<2, b.outputDims, false)
	if err != nil {
		return nil, err
	}
	return clip.GetFeature(b.proj, extracted), nil
}

func (b *Bpe) Close() {
	if b.engine != nil {
		b.engine.Close()
		b.engine = nil
	}
}

func NewBpe(vocab, proj, model string, threads int) (*Bpe, error) {
	tk, err := tokenizer.NewBpeFromFile(vocab)
	if err != nil {
		return nil, err
	}

	pm, err := clip.ReadProjection(proj)
	if err != nil {
		return nil, err
	}

	engine, err := ort.New(model, "text_clip", threads)
	if err != nil {
		return nil, err
	}

	engine.InputDims[0][0] = 1

	return &Bpe{tk, pm, engine, int(engine.InputDims[0][1]), int(engine.OutputDims[0][2])}, nil
}
