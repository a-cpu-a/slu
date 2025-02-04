/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>
#include <vector>
#include <optional>
#include <memory>
#include <variant>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include "Enums.hpp"

//for enums...
#undef FALSE
#undef TRUE

namespace sluaParse
{
	struct Position
	{
		size_t line;
		size_t index;
	};

	struct Scope
	{
		std::vector<struct Variable> variables;
		std::vector<struct Function> functions;

		//Scope* owner;//nullptr -> this is the global scope

		Position start;
		Position end;
	};

	struct Variable
	{
		std::string name;
		//Scope* owner;//nullptr -> error

		Position place;
		//size_t typeId;
	};

	namespace FieldType { struct NONE {}; struct EXPR2EXPR; struct NAME2EXPR; struct EXPR; }

	using Field = std::variant<
		FieldType::NONE,// Here, so variant has a default value (DO NOT USE)

		FieldType::EXPR2EXPR, // "'[' exp ']' = exp"
		FieldType::NAME2EXPR, // "Name = exp"
		FieldType::EXPR       // "exp"
	>;

	// ‘{’ [fieldlist] ‘}’
	using TableConstructor = std::vector<Field>;



	using ExpList = std::vector<struct Expression>;

	struct Block
	{
		std::vector<struct Statement> statList;
		ExpList retExprs;//Special, may contain 0 elements (even with hadReturn)

		//Scope scope;

		Position start;
		Position end;

		bool hadReturn = false;


		Block() = default;
		Block(const Block&) = delete;
		Block(Block&&) = default;
		Block& operator=(Block&&) = default;
	};


	struct Parameter
	{
		std::string name;
		//size_t typeId;
	};

	struct Function
	{
		Position place;
		std::vector<Parameter> params;
		Block block;
		bool hasVarArgParam = false;// do params end with '...'
	};

	namespace ArgsType
	{
		struct EXPLIST { ExpList v; };			// "'(' [explist] ')'"
		struct TABLE { TableConstructor v; };	// "tableconstructor"
		struct LITERAL { std::string v; };		// "LiteralString"
	};
	using Args = std::variant<
		ArgsType::EXPLIST,
		ArgsType::TABLE,
		ArgsType::LITERAL
	>;

	struct ArgFuncCall
	{// funcArgs ::=  [‘:’ Name] args

		std::string funcName;
		Args args;
	};

	namespace LimPrefixExprType
	{
		struct VAR;										// "var"
		struct EXPR;								// "'(' exp ')'"
	}
	using LimPrefixExpr = std::variant<
		LimPrefixExprType::VAR,
		LimPrefixExprType::EXPR
	>;

	struct FuncCall
	{
		std::unique_ptr<LimPrefixExpr> val;
		std::vector<ArgFuncCall> argChain;
	};
	namespace ExprType
	{
		struct NIL {};														// "nil"
		struct FALSE {};													// "false"
		struct TRUE {};														// "true"
		struct NUMERAL { double v; };										// "Numeral"
		struct LITERAL_STRING { std::string v; };							// "LiteralString"
		struct VARARGS {};													// "..."
		struct FUNCTION_DEF { Function v; };								// "functiondef"
		using LIM_PREFIX_EXP = std::unique_ptr<LimPrefixExpr>;				// "prefixexp"
		using FUNC_CALL = FuncCall;											// "functioncall"
		struct TABLE_CONSTRUCTOR { TableConstructor v; };					// "tableconstructor"

		struct MULTI_OPERATION
		{
			std::unique_ptr<struct Expression> first;
			std::vector<std::pair<BinOpType, struct Expression>> extra;//size>=1
		};      // "exp binop exp"

		//struct UNARY_OPERATION{UnOpType,std::unique_ptr<struct Expression>};     // "unop exp"	//Inlined as opt prefix


		struct NUMERAL_I64 { int64_t v; };            // "Numeral"
	}

	using ExprData = std::variant<
		ExprType::NIL,                  // "nil"
		ExprType::FALSE,                // "false"
		ExprType::TRUE,                 // "true"
		ExprType::NUMERAL,				// "Numeral" (e.g., a floating-point number)
		ExprType::NUMERAL_I64,			// "Numeral"
		//ExprType::NUMERAL_U64,		// "Numeral"
		ExprType::LITERAL_STRING,		// "LiteralString"
		ExprType::VARARGS,              // "..." (varargs)
		ExprType::FUNCTION_DEF,			// "functiondef"
		ExprType::LIM_PREFIX_EXP,		// "prefixexp"
		ExprType::FUNC_CALL,			// "prefixexp argsThing {argsThing}"
		ExprType::TABLE_CONSTRUCTOR,	// "tableconstructor"

		ExprType::MULTI_OPERATION		// "exp binop exp {binop exp}"  // added {binop exp}, cuz multi-op

		//ExprType::UNARY_OPERATION,	// "unop exp"
	>;

	struct Expression
	{
		ExprData data;
		Position place;
		UnOpType unOp;//might be NONE

		Expression() = default;
		Expression(const Expression&) = delete;
		Expression(Expression&&) = default;
		Expression& operator=(Expression&&) = default;
	};

	namespace SubVarType
	{
		struct BASE { std::vector<ArgFuncCall> funcCalls; };

		struct NAME :BASE { std::string idx; }; // {funcArgs} ‘.’ Name
		struct EXPR :BASE { Expression idx; };	// {funcArgs} ‘[’ exp ‘]’
	}

	using SubVar = std::variant<
		SubVarType::NAME,
		SubVarType::EXPR
	>;

	namespace BaseVarType
	{
		using NAME = std::string;
		struct EXPR { Expression start; SubVar sub; };
	}
	using BaseVar = std::variant<
		BaseVarType::NAME,
		BaseVarType::EXPR
	>;

	struct Var
	{
		BaseVar base;
		std::vector<SubVar> sub;
	};

	struct AttribName
	{
		std::string name;
		std::string attrib;//empty -> no attrib
	};

	namespace FieldType
	{
		struct EXPR2EXPR { Expression idx; Expression v; };		// "‘[’ exp ‘]’ ‘=’ exp"
		struct NAME2EXPR { std::string idx; Expression v; };	// "Name ‘=’ exp"
		struct EXPR { Expression v; };							// "exp"
	}
	namespace LimPrefixExprType
	{
		struct VAR { Var v; };										// "var"
		struct EXPR { Expression v; };								// "'(' exp ')'"
	}

	using AttribNameList = std::vector<AttribName>;
	using NameList = std::vector<std::string>;

	namespace StatementType
	{
		struct NONE {};// Here, so variant has a default value (DO NOT USE)


		struct SEMICOLON {};									// ";"
		struct ASSIGN { std::vector<Var> vars; ExpList exprs; };// "varlist = explist" //e.size must be > 0
		using FUNC_CALL = FuncCall;								// "functioncall"
		struct LABEL { std::string v; };						// "label"
		struct BREAK { std::string v; };						// "break"
		struct GOTO { std::string v; };							// "goto Name"
		struct DO_BLOCK { Block bl; };							// "do block end"
		struct WHILE_LOOP { Expression cond; Block bl; };		// "while exp do block end"
		struct REPEAT_UNTIL :WHILE_LOOP {};						// "repeat block until exp"

		// "if exp then block {elseif exp then block} [else block] end"
		struct IF_THEN_ELSE
		{
			Expression cond;
			Block bl;
			std::vector<std::pair<Expression, Block>> elseIfs;
			std::optional<Block> elseBlock;
		};
		// "for Name = exp , exp [, exp] do block end"
		struct FOR_LOOP_NUMERIC
		{
			std::string varName;
			Expression start;
			Expression end;//inclusive
			std::optional<Expression> step;
			Block bl;
		};
		// "for namelist in explist do block end"
		struct FOR_LOOP_GENERIC
		{
			NameList varNames;
			ExpList exprs;//size must be > 0
			Block bl;
		};
		struct FUNCTION_DEF { std::string name; Function func; };		// "function funcname funcbody"    //n may contain dots, 1 colon
		struct LOCAL_FUNCTION_DEF :FUNCTION_DEF {};						// "local function Name funcbody" //n may not ^^^
		struct LOCAL_ASSIGN { AttribNameList names; ExpList exprs; };	// "local attnamelist [= explist]" //e.size 0 means "only define, no assign"
	};

	using StatementData = std::variant <
		StatementType::NONE,

		StatementType::SEMICOLON, // ";"

		StatementType::ASSIGN, // "varlist = explist"

		StatementType::LOCAL_ASSIGN, // "local attnamelist [= explist]"

		StatementType::FUNC_CALL, // "functioncall"
		StatementType::LABEL, // "label"
		StatementType::BREAK, // "break"
		StatementType::GOTO, // "goto Name"
		StatementType::DO_BLOCK, // "do block end"
		StatementType::WHILE_LOOP, // "while exp do block end"
		StatementType::REPEAT_UNTIL, // "repeat block until exp"

		StatementType::IF_THEN_ELSE, // "if exp then block {elseif exp then block} [else block] end"

		StatementType::FOR_LOOP_NUMERIC, // "for Name = exp , exp [, exp] do block end"

		StatementType::FOR_LOOP_GENERIC, // "for namelist in explist do block end"

		StatementType::FUNCTION_DEF, // "function funcname funcbody"

		StatementType::LOCAL_FUNCTION_DEF // "local function Name funcbody"
	> ;


	struct Statement
	{
		StatementData data;
		Position place;

		Statement() = default;
		Statement(const Statement&) = delete;
		Statement(Statement&&) = default;
		Statement& operator=(Statement&&) = default;
	};
}