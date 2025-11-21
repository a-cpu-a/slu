/*
** See Copyright Notice inside Include.hpp
*/
#pragma once
// clang-format off
#define Slu_POOLED_ITEMS(...) \
	_X(X,"x") __VA_ARGS__ \
	_X(Y,"y") __VA_ARGS__ \
	_X(Z,"z") __VA_ARGS__ \
	_X(W,"w") __VA_ARGS__ \
\
	_X(I,"i") __VA_ARGS__ \
	_X(J,"j") __VA_ARGS__ \
	_X(K,"k") __VA_ARGS__ \
	_X(L,"l") __VA_ARGS__ \
\
	_X(S,"s") __VA_ARGS__ \
	_X(T,"t") __VA_ARGS__ \
	_X(U,"u") __VA_ARGS__ \
	_X(V,"v") __VA_ARGS__ \
\
	_X(R,"r") __VA_ARGS__ \
	_X(G,"g") __VA_ARGS__ \
	_X(A,"a") __VA_ARGS__ \
	_X(B,"b") __VA_ARGS__ \
	_X(C,"c") __VA_ARGS__ \
	_X(D,"d") __VA_ARGS__ \
	_X(F,"f") __VA_ARGS__ \
	_X(N,"n") __VA_ARGS__ \
	_X(O,"o") __VA_ARGS__ \
\
	_X(U8,"u8") __VA_ARGS__ \
	_X(I8,"i8") __VA_ARGS__ \
	_X(U16,"u16") __VA_ARGS__ \
	_X(I16,"i16") __VA_ARGS__ \
	_X(U32,"u32") __VA_ARGS__ \
	_X(I32,"i32") __VA_ARGS__ \
	_X(U64,"u64") __VA_ARGS__ \
	_X(I64,"i64") __VA_ARGS__ \
	_X(U128,"u128") __VA_ARGS__ \
	_X(I128,"i128") __VA_ARGS__ \
\
	_X(BOOL,"bool") __VA_ARGS__ \
	_X(STR,"str") __VA_ARGS__ \
	_X(CHAR,"char")

#define Slu_STD_ITEMS(...) \
	_X(BOOL,"bool") __VA_ARGS__ \
\
	_X(STR,"str") __VA_ARGS__ \
	_X(CHAR,"char") __VA_ARGS__ \
\
	_X(UNIT,"Unit") __VA_ARGS__ \
\
	_X(U8,"u8") __VA_ARGS__ \
	_X(I8,"i8") __VA_ARGS__ \
	_X(U16,"u16") __VA_ARGS__ \
	_X(I16,"i16") __VA_ARGS__ \
	_X(U32,"u32") __VA_ARGS__ \
	_X(I32,"i32") __VA_ARGS__ \
	_X(U64,"u64") __VA_ARGS__ \
	_X(I64,"i64") __VA_ARGS__ \
	_X(U128,"u128") __VA_ARGS__ \
	_X(I128,"i128")

#define Slu_STD_BOOL_ITEMS(...) \
	_X(TRUE,"true") __VA_ARGS__ \
	_X(FALSE,"false")
// clang-format on