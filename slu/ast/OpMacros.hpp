/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#define Slu_RANGE_OP_TRAIT_NAME Boundable
#define Slu_LTGT_OP_TRAIT_NAME PartialOrd
#define Slu_EQ_OP_TRAIT_NAME PartialEq

#define Slu_UN_OPS(...) \
	_X(NEGATE,Neg,neg) __VA_ARGS__ \
	_X(LOGICAL_NOT,Not,not) __VA_ARGS__ \
\
	_X(RANGE_BEFORE,Slu_RANGE_OP_TRAIT_NAME,rangeMax) __VA_ARGS__ \
\
	_X(ALLOCATE,_Alloc,_alloc) __VA_ARGS__ \
\
	_X(TO_REF,Ref,ref) __VA_ARGS__ \
	_X(TO_REF_MUT,RefMut,refMut) __VA_ARGS__ \
	_X(TO_REF_CONST,RefConst,refConst) __VA_ARGS__ \
	_X(TO_REF_SHARE,RefShare,refShare) __VA_ARGS__ \
\
	_X(TO_PTR,Ptr,ptr) __VA_ARGS__ \
	_X(TO_PTR_MUT, PtrMut,ptrMut) __VA_ARGS__ \
	_X(TO_PTR_CONST,PtrConst,ptrConst) __VA_ARGS__ \
	_X(TO_PTR_SHARE,PtrShare,ptrShare) __VA_ARGS__ \
\
	_X(MUT,MarkMut,markMut)

#define Slu_POST_UN_OPS(...) \
	_X(RANGE_AFTER,Slu_RANGE_OP_TRAIT_NAME,rangeMin) __VA_ARGS__ \
\
	_X(DEREF,Deref,deref) __VA_ARGS__ \
	_X(PROPOGATE_ERR,Try,branch)

#define Slu_BIN_OPS(...) \
	_X(ADD, Add, add) __VA_ARGS__ \
	_X(SUBTRACT, Sub, sub) __VA_ARGS__ \
	_X(MULTIPLY, Mul, mul) __VA_ARGS__ \
	_X(DIVIDE, Div, div) __VA_ARGS__ \
	_X(FLOOR_DIVIDE, FloorDiv, floorDiv) __VA_ARGS__ \
	_X(EXPONENT, Pow, pow) __VA_ARGS__ \
	_X(MODULO, Rem, rem) __VA_ARGS__ \
\
	_X(BITWISE_AND, BitAnd, bitAnd) __VA_ARGS__ \
	_X(BITWISE_XOR, BitXor, bitXor) __VA_ARGS__ \
	_X(BITWISE_OR, BitOr, bitOr) __VA_ARGS__ \
	_X(SHIFT_RIGHT, Shr, shr) __VA_ARGS__ \
	_X(SHIFT_LEFT, Shl, shl) __VA_ARGS__ \
\
	_X(CONCATENATE, Concat, concat) __VA_ARGS__ \
\
	_X(LESS_THAN, Slu_LTGT_OP_TRAIT_NAME, lt) __VA_ARGS__ \
	_X(LESS_EQUAL, Slu_LTGT_OP_TRAIT_NAME, le) __VA_ARGS__ \
	_X(GREATER_THAN, Slu_LTGT_OP_TRAIT_NAME, gt) __VA_ARGS__ \
	_X(GREATER_EQUAL, Slu_LTGT_OP_TRAIT_NAME, ge) __VA_ARGS__ \
	_X(EQUAL, Slu_EQ_OP_TRAIT_NAME, eq) __VA_ARGS__ \
	_X(NOT_EQUAL, Slu_EQ_OP_TRAIT_NAME, ne) __VA_ARGS__ \
\
	_X(LOGICAL_AND, And, and) __VA_ARGS__ \
	_X(LOGICAL_OR, Or, or) __VA_ARGS__ \
\
	_X(ARRAY_MUL, Rep, rep) __VA_ARGS__ \
	_X(RANGE_BETWEEN, Slu_RANGE_OP_TRAIT_NAME, range) __VA_ARGS__ \
	_X(MAKE_RESULT, _MkResult, _mkResult) __VA_ARGS__ \
	_X(UNION, Union, union) __VA_ARGS__ \
	_X(AS, As, asType)
