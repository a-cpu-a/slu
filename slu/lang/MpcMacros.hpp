/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#define Slu_POOLED_ITEMS(_SEP) \
	_X(X,"x") _SEP \
	_X(Y,"y") _SEP \
	_X(Z,"z") _SEP \
	_X(W,"w") _SEP \
\
	_X(I,"i") _SEP \
	_X(J,"j") _SEP \
	_X(K,"k") _SEP \
	_X(L,"l") _SEP \
\
	_X(S,"s") _SEP \
	_X(T,"t") _SEP \
	_X(U,"u") _SEP \
	_X(V,"v") _SEP \
\
	_X(R,"r") _SEP \
	_X(G,"g") _SEP \
	_X(A,"a") _SEP \
	_X(B,"b") _SEP \
	_X(C,"c") _SEP \
	_X(D,"d") _SEP \
	_X(F,"f") _SEP \
	_X(N,"n") _SEP \
	_X(O,"o") _SEP \
\
	_X(U8,"u8") _SEP \
	_X(I8,"i8") _SEP \
	_X(U16,"u16") _SEP \
	_X(I16,"i16") _SEP \
	_X(U32,"u32") _SEP \
	_X(I32,"i32") _SEP \
	_X(U64,"u64") _SEP \
	_X(I64,"i64") _SEP \
	_X(U128,"u128") _SEP \
	_X(I128,"i128") _SEP \
\
	_X(BOOL,"bool") _SEP \
	_X(STR,"str") _SEP \
	_X(CHAR,"char")

#define Slu_STD_ITEMS(_SEP) \
	_X(BOOL,"bool") _SEP \
\
	_X(STR,"str") _SEP \
	_X(CHAR,"char") _SEP \
\
	_X(UNIT,"Unit") _SEP \
\
	_X(U8,"u8") _SEP \
	_X(I8,"i8") _SEP \
	_X(U16,"u16") _SEP \
	_X(I16,"i16") _SEP \
	_X(U32,"u32") _SEP \
	_X(I32,"i32") _SEP \
	_X(U64,"u64") _SEP \
	_X(I64,"i64") _SEP \
	_X(U128,"u128") _SEP \
	_X(I128,"i128")

#define Slu_STD_BOOL_ITEMS(_SEP) \
	_X(TRUE,"true") _SEP \
	_X(FALSE,"false")