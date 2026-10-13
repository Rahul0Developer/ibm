// Package tokenizer converts raw UTF-8 text into fixed-width INT64 token-id
// windows compatible with the exported transformer ONNX graph. It implements a
// deterministic WordPiece-style subword split over a hashed 30k vocabulary —
// production deployments should swap in the exact HF vocab.txt for
// BAAI/bge-small-en-v1.5 behind this same interface.
package tokenizer

const (
	// Special tokens mirroring the BERT/Roberta family.
	IDPad int64 = 0
	IDCls int64 = 101
	IDSep int64 = 102
	IDUnk int64 = 100

	vocabSize = 30000
	offset    = 200 // first hashed word-piece id (specials + basic vocab live below)
	maxWord   = 50  // cap per-word splits so adversarial input can't blow up CPU
)

// Tokenizer is stateless and safe for concurrent use.
type Tokenizer struct{}

// New returns a ready-to-use tokenizer.
func New() *Tokenizer { return &Tokenizer{} }

// Encode maps text to token ids, then pads/truncates to exactly maxLen.
// Window layout: [CLS] pieces... [SEP] [PAD...]
func (t *Tokenizer) EncodePadded(text string, maxLen int) []int64 {
	ids := make([]int64, 0, maxLen)
	ids = append(ids, IDCls)
	for _, w := range words(text) {
		for _, p := range wordPieces(w) {
			if len(ids) >= maxLen-1 {
				break
			}
			ids = append(ids, hashPiece(p))
		}
	}
	ids = append(ids, IDSep)
	for len(ids) < maxLen {
		ids = append(ids, IDPad)
	}
	return ids[:maxLen]
}

// words lowercases and splits on whitespace + common punctuation boundaries.
func words(s string) []string {
	out := []string{}
	cur := make([]rune, 0, 16)
	flush := func() {
		if len(cur) > 0 {
			out = append(out, string(cur))
			cur = cur[:0]
		}
	}
	for _, r := range s {
		switch {
		case r >= 'A' && r <= 'Z':
			cur = append(cur, r+32)
		case r == ' ' || r == '\t' || r == '\n' || r == '\r':
			flush()
		case isPunct(r):
			flush()
			cur = append(cur, r)
			flush()
		default:
			cur = append(cur, r)
		}
	}
	flush()
	return out
}

func isPunct(r rune) bool {
	switch r {
	case '.', ',', ':', ';', '!', '?', '"', '\'', '(', ')', '[', ']', '{', '}', '-', '/':
		return true
	}
	return false
}

// wordPieces greedily peels the longest matching prefix (WordPiece greedy
// match). Without a shipped vocab file every prefix "matches" by hash, so we
// keep whole words unless they exceed maxWord runes.
func wordPieces(w string) []string {
	runes := []rune(w)
	if len(runes) <= maxWord {
		return []string{w}
	}
	pieces := make([]string, 0, len(runes)/maxWord+1)
	for i := 0; i < len(runes); i += maxWord {
		end := i + maxWord
		if end > len(runes) {
			end = len(runes)
		}
		pieces = append(pieces, string(runes[i:end]))
	}
	return pieces
}

// hashPiece FNV-1a maps a piece into the usable id range deterministically.
func hashPiece(p string) int64 {
	var h uint64 = 14695981039346656037
	for i := 0; i < len(p); i++ {
		h ^= uint64(p[i])
		h *= 1099511628211
	}
	return int64(offset + h%(vocabSize-uint64(offset)))
}
