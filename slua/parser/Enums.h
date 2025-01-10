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

	namespace ExprType
	{
		struct NIL {};													// "nil"
		struct FALSE {};												// "false"
		struct TRUE {};													// "true"
		struct NUMERAL { double v; };									// "Numeral"
		struct LITERAL_STRING{ std::string v; };						// "LiteralString"
		struct VARARGS{};												// "..."
		struct FUNCTION_DEF { Function v; };							// "functiondef"
		struct PREFIX_EXP { std::unique_ptr<struct PrefixExpr> v; };	// "prefixexp"
		struct TABLE_CONSTRUCTOR { TableConstructor v; };				// "tableconstructor"

		struct BINARY_OPERATION 
		{
			BinOpType t;
			std::unique_ptr<struct Expression> l;
			std::unique_ptr<struct Expression> r;
		};      // "exp binop exp"

		//struct UNARY_OPERATION{UnOpType,std::unique_ptr<struct Expression>};     // "unop exp"	//Inlined as opt prefix


		struct NUMERAL_I64{ int64_t v; };            // "Numeral"
	}

	namespace VarType
	{
		struct INDEX_STR { std::unique_ptr<struct PrefixExpr> var; std::string idx; };	// "prefixexp.Name"
		struct INDEX { std::unique_ptr<struct PrefixExpr> var; Expression idx; };		// "prefixexp [ exp ]"
		struct VAR_NAME { std::string var; };											// "Name"
	}

	namespace PrefixExprType
	{
		struct VAR { Var v; };										// "var"
		struct FUNC_CALL { std::unique_ptr<struct FuncCall> c; };	// "functioncall"
		struct EXPR { Expression e; };								// "'(' exp ')'"
	}
	namespace ArgsType
	{
		struct EXPLIST { ExpList v; };			// "'(' [explist] ')'"
		struct TABLE { TableConstructor v; };	// "tableconstructor"
		struct LITERAL { std::string v; };		// "LiteralString"
	};
	namespace FieldType
	{
		struct EXPR2EXPR { Expression i; Expression v; };	// "‘[’ exp ‘]’ ‘=’ exp"
		struct NAME2EXPR { std::string i; Expression v; };	// "Name ‘=’ exp"
		struct EXPR { Expression v; };						// "exp"
	};
}