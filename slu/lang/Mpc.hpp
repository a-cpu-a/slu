/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <cstdint>
#include <array>
#include <slu/lang/BasicState.hpp>

namespace slu::mpc
{
	using namespace std::string_view_literals;

	template<size_t N>
	struct MpcMp
	{
		std::array<std::string_view,N> mp;
		lang::ModPathId id;

		constexpr size_t idx() const {
			return id.id;
		}
		constexpr operator lang::ModPathId() const {
			return id;
		}
	};
	struct MpcItm
	{
		lang::MpItmIdV<true> itm;
		std::string_view name;

		constexpr operator lang::MpItmIdV<true>() const {
			return itm;
		}
	};

	//Warning: compiler doesnt care what N is!!
	template<size_t N>
	consteval auto mkMp(std::array<std::string_view, N> path,size_t id) {
		return MpcMp{ .mp=path,.id = {id}};
	}
	consteval MpcItm mkItm(lang::ModPathId mp, size_t localId, std::string_view str) {
		return MpcItm{ .itm = {lang::LocalObjId{localId},mp},.name = str };
	}
	constexpr auto MP_UNKNOWN = mkMp<1>({""sv}, 0);
	constexpr auto MP_POOLED = MP_UNKNOWN;
	constexpr auto MP_STD = mkMp<1>({"std"sv}, 1);
	constexpr auto MP_STD_BOOL = mkMp<2>({"std"sv,"bool"sv}, 2);
	constexpr size_t MP_COUNT = 3;

	static_assert(MP_UNKNOWN.id.id == 0);
	static_assert(MP_STD.id.id == 1);

#define _Slu_DEF_ITM(ARG_MP,_VAR,_STR) \
	constexpr ::slu::mpc::MpcItm \
	ARG_MP##_##_VAR = ::slu::mpc::mkItm( \
			::slu::mpc::MP_##ARG_MP.id, \
			__COUNTER__ - (::slu::mpc::_##ARG_MP##_COUNTER_START+1), \
			_STR##sv \
		)

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

#define _Slu_MK_COUNTER_RUN(ARG_MP) \
	constexpr size_t _##ARG_MP##_COUNTER_START = __COUNTER__; \
	Slu_##ARG_MP##_ITEMS(;); \
	constexpr size_t MP_ITM_COUNT_##ARG_MP = __COUNTER__ - (_##ARG_MP##_COUNTER_START +1)

#define _X(_VAR,_STR) _Slu_DEF_ITM(POOLED,_VAR,_STR)
	_Slu_MK_COUNTER_RUN(POOLED);
#undef _X
#define _X(_VAR,_STR) _Slu_DEF_ITM(STD,_VAR,_STR)
	_Slu_MK_COUNTER_RUN(STD);
#undef _X
#define _X(_VAR,_STR) _Slu_DEF_ITM(STD_BOOL,_VAR,_STR)
	_Slu_MK_COUNTER_RUN(STD_BOOL);
#undef _X

	static_assert(STD_BOOL.itm.id.val == 0);
	static_assert(STD_STR.itm.id.val == 1);
	static_assert(STD_CHAR.itm.id.val == 2);

	static_assert(MP_ITM_COUNT_STD_BOOL == 2);
}