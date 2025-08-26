module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <cstdint>

#include <slu/ast/OpMacros.hpp>
export module slu.ast.enums;

namespace slu::ast
{
	export enum class OptSafety : uint8_t
	{
		DEFAULT,
		SAFE,
		UNSAFE
	};

	export enum class BinOpType : uint8_t
	{
		NONE,
#define _X(E,T,M) E
		Slu_BIN_OPS(, ),

		//Not a real op, just the amount of binops
		// -1 because NONE, +1 cuz total size
		ENUM_SIZE = AS - 1+1
	};

	export enum class UnOpType : uint8_t
	{
		NONE,

		Slu_UN_OPS(, ),

		//Not a real op, just the amount of unops
		//== MUT because NONE isnt a real op.
		ENUM_SIZE = MUT
	};

	export enum class PostUnOpType : uint8_t
	{
		NONE,

		Slu_POST_UN_OPS(, ),

		//Not a real op, just the amount of post unops
		// -1 because NONE, +1 cuz total size
		ENUM_SIZE = PROPOGATE_ERR+1-1
	};
}
