module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <string_view>
#include <cstdint>
#include <array>

#include <slu/lang/MpcMacros.hpp>
export module slu.lang.mpc;

import slu.lang.basic_state;

namespace slu::mpc
{
	using namespace std::string_view_literals;

	export template<size_t N>
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
	export struct MpcItm
	{
		lang::MpItmId itm;
		std::string_view name;

		constexpr operator lang::MpItmId() const {
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
	export constexpr auto MP_UNKNOWN = mkMp<1>({""sv}, 0);
	export constexpr auto MP_POOLED = MP_UNKNOWN;
	export constexpr auto MP_STD = mkMp<1>({"std"sv}, 1);
	export constexpr auto MP_STD_BOOL = mkMp<2>({"std"sv,"bool"sv}, 2);
	export constexpr size_t MP_COUNT = 3;

	static_assert(MP_UNKNOWN.id.id == 0);
	static_assert(MP_STD.id.id == 1);

#define _Slu_DEF_ITM(ARG_MP,_VAR,_STR) \
	export constexpr ::slu::mpc::MpcItm \
	ARG_MP##_##_VAR = ::slu::mpc::mkItm( \
			::slu::mpc::MP_##ARG_MP.id, \
			__COUNTER__ - (::slu::mpc::_##ARG_MP##_COUNTER_START+1), \
			_STR##sv \
		)


#define _Slu_MK_COUNTER_RUN(ARG_MP) \
	constexpr size_t _##ARG_MP##_COUNTER_START = __COUNTER__; \
	Slu_##ARG_MP##_ITEMS(;); \
	export constexpr size_t MP_ITM_COUNT_##ARG_MP = __COUNTER__ - (_##ARG_MP##_COUNTER_START +1)

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