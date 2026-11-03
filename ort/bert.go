package ort

import (
	"unsafe"

	"github.com/oxzjh/ort"
	"github.com/oxzjh/tokenizer"
	"github.com/oxzjh/vector"
)

type Bert struct {
	tk         *tokenizer.BertTokenizer
	engine     *ort.Ort
	inputDims  int
	outputDims int
}

func (b *Bert) Extract(text string) ([]float32, error) {
	tokens := b.tk.Encode(text, true)
	result, err := b.engine.Infer(unsafe.Pointer(&tokens.GetInt64(b.inputDims, 0)[0]), b.inputDims<<3, ort.TypeInt64)
	if err != nil {
		return nil, err
	}
	defer result.Free()
	feature, err := result.GetFloat32(0, b.outputDims, true)
	if err != nil {
		return nil, err
	}
	vector.NormalizeSelf(feature)
	return feature, nil
}

func (b *Bert) Close() {
	if b.engine != nil {
		b.engine.Close()
		b.engine = nil
	}
}

func NewBert(vocab, model string, threads int) (*Bert, error) {
	tk, err := tokenizer.NewBertFromFile(vocab, tokenizer.DefaultMaxWordChars)
	if err != nil {
		return nil, err
	}
	engine, err := ort.New(model, "text_clip", threads)
	if err != nil {
		return nil, err
	}
	return &Bert{tk, engine, int(engine.InputDims[0][1]), int(engine.OutputDims[0][1])}, nil
}
