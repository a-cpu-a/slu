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
	namespace StatementType
	{
		struct SEMICOLON {};									// ";"
		struct ASSIGN { AttribNameList n; ExpList e; };         // "varlist = explist" //e.size must be > 0
		struct FUNC_CALL { FuncCall v; };						// "functioncall"
		struct LABEL {};										// "label"
		struct BREAK { std::string v; };						// "break"
		struct GOTO { std::string v; };							// "goto Name"
		struct DO_BLOCK { Block v; };							// "do block end"
		struct WHILE_LOOP { Expression c; Block b; };           // "while exp do block end"
		struct REPEAT_UNTIL :WHILE_LOOP {};						// "repeat block until exp"

		// "if exp then block {elseif exp then block} [else block] end"
		struct IF_THEN_ELSE 
		{
			Expression c; 
			Block b;
			std::vector<std::pair<Expression, Block>> elseIfs;
			std::optional<Block> elseBlock;
		};            
		// "for Name = exp , exp [, exp] do block end"
		struct FOR_LOOP_NUMERIC 
		{
			std::string varN;
			Expression start;
			Expression end;//inclusive
			std::optional<Expression> step;
			Block b;
		};
		// "for namelist in explist do block end"
		struct FOR_LOOP_GENERIC {
			NameList varNs;
			ExpList eList;//size must be > 0
			Block b;
		};
		struct FUNCTION_DEF { std::string n; Function f; };// "function funcname funcbody"    //n may contain dots, 1 colon
		struct LOCAL_FUNCTION_DEF :FUNCTION_DEF {};        // "local function Name funcbody" //n may not ^^^
		struct LOCAL_ASSIGN :ASSIGN {};			   // "local attnamelist [= explist]" //e.size 0 means "only define, no assign"
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