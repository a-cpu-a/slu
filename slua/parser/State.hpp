/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>
#include <vector>
#include <optional>
#include <memory>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <sus/choice/choice.h>

#include "Enums.h"

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

		Scope* owner;//nullptr -> this is the global scope

		Position start;
		Position end;
	};

	struct Variable
	{
		std::string name;
		Scope* owner;//nullptr -> error

		Position place;
		//size_t typeId;
	};

	// ‘{’ [fieldlist] ‘}’
	using TableConstructor = std::vector<struct Field>;



	using ExpList = std::vector<struct Expression>;

	struct Block
	{
		std::vector<struct Statement> statList;
		ExpList ret;//if empty, then no return, see hadEmptyReturn

		Scope scope;

		Position start;
		Position end;

		bool hadEmptyReturn = false;
	};


	struct Parameter
	{
		std::string name;
		//size_t typeId;
	};

	struct Function : Variable
	{
		std::vector<Parameter> params;
		Block block;
		bool hasVarArgParam = false;// does params end with '...'
	};

	using ExprData = sus::Choice<sus_choice_types(
		(ExprType::NIL, void),                    // "nil"
		(ExprType::FALSE, void),                  // "false"
		(ExprType::TRUE, void),                   // "true"
		(ExprType::NUMERAL, double),              // "Numeral" (e.g., a floating-point number)
		(ExprType::NUMERAL_I64, int64_t),         // "Numeral"
		(ExprType::NUMERAL_U64, uint64_t),        // "Numeral"
		(ExprType::LITERAL_STRING, std::string),  // "LiteralString"
		(ExprType::VARARGS, void),                // "..." (varargs)
		(ExprType::FUNCTION_DEF, Function),       // "functiondef"
		(ExprType::PREFIX_EXP, std::unique_ptr<struct PrefixExpr>),      // "prefixexp"
		(ExprType::TABLE_CONSTRUCTOR, TableConstructor),				// "tableconstructor"

		(ExprType::BINARY_OPERATION, // "exp binop exp"
			BinOpType, 
			std::unique_ptr<struct Expression>, 
			std::unique_ptr<struct Expression>),

		(ExprType::UNARY_OPERATION,// "unop exp"
			UnOpType,
			std::unique_ptr<struct Expression>)
	)>;

	struct Expression
	{
		ExprData data;
		Position place;
	};

	using Var = sus::Choice<sus_choice_types(
		(VarType::EXP_INDEX, std::unique_ptr<struct PrefixExpr>, Expression),
		(VarType::EXP_INDEX_STR, std::unique_ptr<struct PrefixExpr>, std::string),
		(VarType::VAR_NAME, std::string)
	)>;

	using PrefixExpr = sus::Choice<sus_choice_types(
		(PrefixExprType::VAR, Var),
		(PrefixExprType::FUNC_CALL, std::unique_ptr<struct FuncCall>),
		(PrefixExprType::EXPR, Expression)
	)>;

	using Field = sus::Choice<sus_choice_types(
		(FieldType::EXPR2EXPR, Expression, Expression),  // "[ exp ] = exp"
		(FieldType::NAME2EXPR, std::string, Expression), // "Name = exp"
		(FieldType::EXPR, Expression)                    // "exp"
	)>;

	using Args = sus::Choice<sus_choice_types(
		(ArgsType::EXPLIST, ExpList),
		(ArgsType::TABLE, TableConstructor),
		(ArgsType::LITERAL, std::string)
	)>;

	struct FuncCall
	{
		PrefixExpr val;
		std::optional<std::string> funcName;// for abc:___()
		Args args;
	};

	struct AttribName
	{
		std::string name;
		std::optional<std::string> attrib;
	};

	using AttribNameList = std::vector<AttribName>;
	using NameList = std::vector<std::string>;

	using StatementData = sus::Choice < sus_choice_types(
		(StatementType::SEMICOLON,void), // ";"

		(StatementType::ASSIGN, // "varlist = explist"
			AttribNameList,
			ExpList),//size must be > 0

		(StatementType::LOCAL_ASSIGN, // "local attnamelist [= explist]"
			AttribNameList,
			ExpList),//size 0 means "only define, no assign"

		(StatementType::FUNC_CALL, FuncCall), // "functioncall"
		(StatementType::LABEL, std::string), // "label"
		(StatementType::BREAK, void), // "break"
		(StatementType::GOTO, std::string), // "goto Name"
		(StatementType::DO_BLOCK, Block), // "do block end"
		(StatementType::WHILE_LOOP, Expression, Block), // "while exp do block end"
		(StatementType::REPEAT_UNTIL, Block, Expression), // "repeat block until exp"

		(StatementType::IF_THEN_ELSE, // "if exp then block {elseif exp then block} [else block] end"
			Expression,               // Initial condition
			Block,                    // Initial block
			std::vector<std::pair<Expression, Block>>, // {elseif exp then block}
			std::optional<Block>),    // [else block]

		(StatementType::FOR_LOOP_NUMERIC, // "for Name = exp , exp [, exp] do block end"
			std::string,              // Name
			Expression,               // Start
			Expression,               // End (inclusive)
			std::optional<Expression>,// Step
			Block),                   // Block

		(StatementType::FOR_LOOP_GENERIC, // "for namelist in explist do block end"
			NameList,                  // namelist
			ExpList,                   // explist !!! size must be > 0
			Block),                    // Block

		(StatementType::FUNCTION_DEF, // "function funcname funcbody"
			std::string,              // funcname, may contain dots, 1 colon
			Function),                // funcbody

		(StatementType::LOCAL_FUNCTION_DEF, // "local function Name funcbody"
			std::string,              // Name
			Function)                 // funcbody
	)>;


	struct Statement
	{
		StatementData data;
		Position place;
	};
}