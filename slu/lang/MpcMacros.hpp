/*
    Slu language compiler, a computer program compiler.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
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