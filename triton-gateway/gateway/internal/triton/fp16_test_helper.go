package triton

import (
	"encoding/binary"
	"math"
)

// float32ToFP16Bytes widens/narrows a float32 to IEEE binary16 little-endian
// bytes. Used by the in-process test backend to emit realistic FP16 payloads;
// production traffic decodes Triton's own FP16 buffers via unpackFloat16.
func float32ToFP16Bytes(f float32) []byte {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int32((bits>>23)&0xff) - 127
	mant := bits & 0x7fffff

	if exp == 128 { // inf / nan
		if mant == 0 {
			return le16(sign | 0x7c00)
		}
		return le16(sign | 0x7e00)
	}
	if exp >= 16 { // overflow to inf
		return le16(sign | 0x7c00)
	}
	if exp >= -14 {
		return le16(sign | uint16(exp+15)<<10 | uint16(mant>>13))
	}
	if exp < -24 { // underflow to zero
		return le16(sign)
	}
	// subnormal
	mant |= 0x800000
	shift := uint(-exp + 13)
	half := mant >> shift
	// round-to-nearest-even on the dropped bit
	if mant&(1<<(shift-1)) != 0 && half&1 == 0 {
		half++
	}
	return le16(sign | uint16(half>>13))
}

func le16(v uint16) []byte {
	b := make([]byte, 2)
	binary.LittleEndian.PutUint16(b, v)
	return b
}

func fp32bits(f float32) uint32 { return math.Float32bits(f) }
