/*
    A program file.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This file is part of Slu-c.

    Slu-c is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Slu-c is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with Slu-c.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
*/
#pragma once

#define Slu_RANGE_OP_TRAIT_NAME Boundable
#define Slu_LTGT_OP_TRAIT_NAME PartialOrd
#define Slu_EQ_OP_TRAIT_NAME PartialEq

#define Slu_UN_OPS(...)                                                       \
	_X(NEG, Neg, neg)                                                         \
	__VA_ARGS__ _X(NOT, Not, not) __VA_ARGS__                                 \
                                                                              \
	    _X(RANGE_BEFORE, Slu_RANGE_OP_TRAIT_NAME, rangeMax)                   \
	__VA_ARGS__                                                               \
                                                                              \
	_X(ALLOC, _Alloc, _alloc)                                                 \
	__VA_ARGS__                                                               \
                                                                              \
	_X(REF, Ref, ref)                                                         \
	__VA_ARGS__                                                               \
	_X(REF_MUT, RefMut, refMut) __VA_ARGS__ _X(REF_CONST, RefConst, refConst) \
	__VA_ARGS__                                                               \
	_X(REF_SHARE, RefShare, refShare)                                         \
	__VA_ARGS__                                                               \
                                                                              \
	_X(PTR, Ptr, ptr)                                                         \
	__VA_ARGS__                                                               \
	_X(PTR_MUT, PtrMut, ptrMut) __VA_ARGS__ _X(PTR_CONST, PtrConst, ptrConst) \
	__VA_ARGS__                                                               \
	_X(PTR_SHARE, PtrShare, ptrShare)                                         \
	__VA_ARGS__                                                               \
                                                                              \
	_X(SLICIFY, Slicify, slicify)                                             \
	__VA_ARGS__                                                               \
                                                                              \
	_X(MARK_MUT, MarkMut, markMut)

#define Slu_POST_UN_OPS(...)                           \
	_X(RANGE_AFTER, Slu_RANGE_OP_TRAIT_NAME, rangeMin) \
	__VA_ARGS__                                        \
                                                       \
	_X(DEREF, Deref, deref)                            \
	__VA_ARGS__                                        \
	_X(TRY, Try, branch)

#define Slu_BIN_OPS(...)                                             \
	_X(ADD, Add, add) __VA_ARGS__ _X(SUB, Sub, sub)                  \
	__VA_ARGS__                                                      \
	_X(MUL, Mul, mul) __VA_ARGS__ _X(DIV, Div, div)                  \
	__VA_ARGS__                                                      \
	_X(FLOOR_DIV, FloorDiv, floorDiv) __VA_ARGS__ _X(EXP, Pow, pow)  \
	__VA_ARGS__                                                      \
	_X(REM, Rem, rem)                                                \
	__VA_ARGS__                                                      \
                                                                     \
	_X(BIT_AND, BitAnd, bitAnd)                                      \
	__VA_ARGS__                                                      \
	_X(BIT_XOR, BitXor, bitXor) __VA_ARGS__ _X(BIT_OR, BitOr, bitOr) \
	__VA_ARGS__                                                      \
	_X(SHR, Shr, shr) __VA_ARGS__ _X(SHL, Shl, shl)                  \
	__VA_ARGS__                                                      \
                                                                     \
	_X(CONCAT, Concat, concat)                                       \
	__VA_ARGS__                                                      \
                                                                     \
	_X(LT, Slu_LTGT_OP_TRAIT_NAME, lt)                               \
	__VA_ARGS__                                                      \
	_X(LE, Slu_LTGT_OP_TRAIT_NAME, le)                               \
	__VA_ARGS__ _X(GT, Slu_LTGT_OP_TRAIT_NAME, gt)                   \
	__VA_ARGS__                                                      \
	_X(GE, Slu_LTGT_OP_TRAIT_NAME, ge)                               \
	__VA_ARGS__ _X(EQ, Slu_EQ_OP_TRAIT_NAME, eq)                     \
	__VA_ARGS__                                                      \
	_X(NE, Slu_EQ_OP_TRAIT_NAME, ne)                                 \
	__VA_ARGS__                                                      \
                                                                     \
	_X(AND, And, and)                                                \
	__VA_ARGS__ _X(OR, Or, or) __VA_ARGS__                           \
                                                                     \
	    _X(REP, Rep, rep)                                            \
	__VA_ARGS__                                                      \
	_X(RANGE, Slu_RANGE_OP_TRAIT_NAME, range)                        \
	__VA_ARGS__ _X(MK_RESULT, _MkResult, _mkResult)                  \
	__VA_ARGS__                                                      \
	_X(UNION, Union, union) __VA_ARGS__ _X(AS, As, asType)
