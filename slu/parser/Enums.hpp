﻿/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

namespace slu::parse
{
	enum class OptSafety : uint8_t
	{
		DEFAULT,
		SAFE,
		UNSAFE
	};

	enum class BinOpType : uint8_t
	{
		NONE,

		ADD,            // "+"
		SUBTRACT,       // "-"
		MULTIPLY,       // "*"
		DIVIDE,         // "/"
		FLOOR_DIVIDE,   // "//"
		EXPONENT,       // "^"
		MODULO,         // "%"
		BITWISE_AND,    // "&"
		BITWISE_XOR,    // "~"
		BITWISE_OR,     // "|"
		SHIFT_RIGHT,    // ">>"
		SHIFT_LEFT,     // "<<"
		CONCATENATE,    // ".." or "++"
		LESS_THAN,      // "<"
		LESS_EQUAL,     // "<="
		GREATER_THAN,   // ">"
		GREATER_EQUAL,  // ">="
		EQUAL,          // "=="
		NOT_EQUAL,      // "~="
		LOGICAL_AND,    // "and"
		LOGICAL_OR,     // "or"

		// Slu

		ARRAY_MUL,// "**"
		RANGE_BETWEEN,	// ".."
		MAKE_RESULT,	// "~~"

		//Not a real op, just the amount of binops
		// -1 because NONE, +1 cuz total size
		ENUM_SIZE = MAKE_RESULT - 1+1
	};

	enum class UnOpType : uint8_t
	{
		NONE,

		NEGATE,        // "-"
		LOGICAL_NOT,   // "not"
		LENGTH,        // "#"
		BITWISE_NOT,   // "~"

		//Slu

		RANGE_BEFORE,	// ".."

		ALLOCATE,		// "alloc"

		TO_REF,			// "&"
		TO_REF_MUT,		// "&mut"
		TO_REF_CONST,	// "&const"
		TO_REF_SHARE,	// "&share"

		TO_PTR,			// "*"
		TO_PTR_MUT,		// "*mut"
		TO_PTR_CONST,	// "*const"
		TO_PTR_SHARE,	// "*share"


		//Pseudo, only for type prefixes
		MUT,			// "mut"

		//Not a real op, just the amount of unops
		//equal to mut, cuz 'mut' isnt a real unop
		// -1 because NONE
		ENUM_SIZE = MUT-1
	};

	enum class PostUnOpType : uint8_t
	{
		NONE,

		RANGE_AFTER,	// ".."

		DEREF,			// ".*"

		PROPOGATE_ERR,	// "?"

		//Not a real op, just the amount of post unops
		// -1 because NONE, +1 cuz total size
		ENUM_SIZE = PROPOGATE_ERR+1-1
	};
}
