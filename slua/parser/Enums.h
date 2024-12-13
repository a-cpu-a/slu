/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

namespace sluaParse
{
	enum class StatementType : uint8_t
	{
		SEMICOLON,              // ";"
		ASSIGN,                 // "varlist = explist"
		FUNC_CALL,              // "functioncall"
		LABEL,                  // "label"
		BREAK,                  // "break"
		GOTO,                   // "goto Name"
		DO_BLOCK,               // "do block end"
		WHILE_LOOP,             // "while exp do block end"
		REPEAT_UNTIL,           // "repeat block until exp"
		IF_THEN_ELSE,           // "if exp then block {elseif exp then block} [else block] end"
		FOR_LOOP_NUMERIC,       // "for Name = exp , exp [, exp] do block end"
		FOR_LOOP_GENERIC,       // "for namelist in explist do block end"
		FUNCTION_DEF,           // "function funcname funcbody"
		LOCAL_FUNCTION_DEF,     // "local function Name funcbody"
		LOCAL_ASSIGN            // "local attnamelist [= explist]"
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
		CONCATENATE,    // ".."
		LESS_THAN,      // "<"
		LESS_EQUAL,     // "<="
		GREATER_THAN,   // ">"
		GREATER_EQUAL,  // ">="
		EQUAL,          // "=="
		NOT_EQUAL,      // "~="
		LOGICAL_AND,    // "and"
		LOGICAL_OR      // "or"
	};

	enum class UnOpType : uint8_t
	{
		NONE,

		NEGATE,        // "-"
		LOGICAL_NOT,   // "not"
		LENGTH,        // "#"
		BITWISE_NOT    // "~"
	};

	enum class ExprType : uint8_t
	{
		NIL,                   // "nil"
		FALSE,                 // "false"
		TRUE,                  // "true"
		NUMERAL,               // "Numeral"
		LITERAL_STRING,        // "LiteralString"
		VARARGS,               // "..."
		FUNCTION_DEF,          // "functiondef"
		PREFIX_EXP,            // "prefixexp"
		TABLE_CONSTRUCTOR,     // "tableconstructor"
		BINARY_OPERATION,      // "exp binop exp"
		UNARY_OPERATION,       // "unop exp"


		NUMERAL_I64,           // "Numeral"
		NUMERAL_U64,           // "Numeral"
	};

	enum class VarType : uint8_t
	{
		VAR_NAME,	   // "Name"
		EXP_INDEX,     // "prefixexp [ exp ]"
		EXP_INDEX_STR  // "prefixexp.Name"
	};

	enum class PrefixExprType : uint8_t
	{
		VAR,			// "var"
		FUNC_CALL,      // "functioncall"
		EXPR 			// "'(' exp ')'"
	};
	enum class ArgsType : uint8_t
	{
		EXPLIST,		// "'(' [explist] ')'"
		TABLE,	        // "tableconstructor"
		LITERAL 		// "LiteralString"
	};
	enum class FieldType : uint8_t
	{
		EXPR2EXPR,		// "‘[’ exp ‘]’ ‘=’ exp"
		NAME2EXPR,	    // "Name ‘=’ exp"
		EXPR 			// "exp"
	};
}