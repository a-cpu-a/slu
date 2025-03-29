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
#include "UnOpList.hpp"
#include "Input.hpp"

//for enums...
#undef FALSE
#undef TRUE

namespace sluaParse
{
	template<AnyCfgable CfgT, class T, class SlT>
	using SelectT = std::conditional_t<CfgT::settings() & sluaSyn, SlT, T>;
	template<AnyCfgable CfgT, template<bool> class T>
	using SelV = T<false>;//CfgT::settings() & sluaSyn
	template<bool isSlua, class T, class SlT>
	using SelectBoolT = std::conditional_t<isSlua, SlT, T>;



	//Forward declare

	template<bool isSlua> struct StatementV;
	template<AnyCfgable CfgT> using Statement = SelV<CfgT, StatementV>;







	namespace FieldType { struct NONE {}; struct EXPR2EXPR; struct NAME2EXPR; struct EXPR; }

	using LuaField = std::variant<
		FieldType::NONE,// Here, so variant has a default value (DO NOT USE)

		FieldType::EXPR2EXPR, // "'[' exp ']' = exp"
		FieldType::NAME2EXPR, // "Name = exp"
		FieldType::EXPR       // "exp"
	>;

	template<AnyCfgable CfgT>
	using Field = SelectT<CfgT, LuaField, LuaField>;

	// ‘{’ [fieldlist] ‘}’
	using LuaTableConstructor = std::vector<LuaField>;

	template<AnyCfgable CfgT>
	using TableConstructor = SelectT<CfgT, LuaTableConstructor, LuaTableConstructor>;



	using LuaExpList = std::vector<struct LuaExpression>;

	template<AnyCfgable CfgT>
	using ExpList = SelectT<CfgT, LuaExpList, LuaExpList>;

	template<bool isSlua>
	struct BlockV
	{
		std::vector<StatementV<isSlua>> statList;
		LuaExpList retExprs;//Special, may contain 0 elements (even with hadReturn)

		//Scope scope;

		Position start;
		Position end;

		bool hadReturn = false;


		BlockV() = default;
		BlockV(const BlockV&) = delete;
		BlockV(BlockV&&) = default;
		BlockV& operator=(BlockV&&) = default;
	};

	template<AnyCfgable CfgT>
	using Block = SelV<CfgT, BlockV>;


	struct LuaParameter
	{
		std::string name;
		//size_t typeId;
	};

	template<AnyCfgable CfgT>
	using Parameter = SelectT<CfgT, LuaParameter, LuaParameter>;

	struct LuaFunction
	{
		std::vector<LuaParameter> params;
		BlockV<false> block;
		bool hasVarArgParam = false;// do params end with '...'
	};

	template<AnyCfgable CfgT>
	using Function = SelectT<CfgT, LuaFunction, LuaFunction>;

	namespace ArgsType
	{
		struct EXPLIST { LuaExpList v; };			// "'(' [explist] ')'"
		struct TABLE { LuaTableConstructor v; };	// "tableconstructor"
		struct LITERAL { std::string v; };		// "LiteralString"
	};
	using LuaArgs = std::variant<
		ArgsType::EXPLIST,
		ArgsType::TABLE,
		ArgsType::LITERAL
	>;

	template<AnyCfgable CfgT>
	using Args = SelectT<CfgT, LuaArgs, LuaArgs>;

	struct LuaArgFuncCall
	{// funcArgs ::=  [‘:’ Name] args

		std::string funcName;//If empty, then no colon needed. Only used for ":xxx"
		LuaArgs args;
	};

	template<AnyCfgable CfgT>
	using ArgFuncCall = SelectT<CfgT, LuaArgFuncCall, LuaArgFuncCall>;


	namespace LimPrefixExprType
	{
		struct VAR;		// "var"
		struct EXPR;	// "'(' exp ')'"
	}
	using LuaLimPrefixExpr = std::variant<
		LimPrefixExprType::VAR,
		LimPrefixExprType::EXPR
	>;

	template<AnyCfgable CfgT>
	using LimPrefixExpr = SelectT<CfgT, LuaLimPrefixExpr, LuaLimPrefixExpr>;

	struct LuaFuncCall
	{
		std::unique_ptr<LuaLimPrefixExpr> val;
		std::vector<LuaArgFuncCall> argChain;
	};

	template<AnyCfgable CfgT>
	using FuncCall = SelectT<CfgT, LuaFuncCall, LuaFuncCall>;


	namespace ExprType
	{
		struct NIL {};											// "nil"
		struct FALSE {};										// "false"
		struct TRUE {};											// "true"
		struct NUMERAL { double v; };							// "Numeral"
		struct LITERAL_STRING { std::string v; };				// "LiteralString"
		struct VARARGS {};										// "..."
		struct FUNCTION_DEF { LuaFunction v; };					// "functiondef"
		using LIM_PREFIX_EXP = std::unique_ptr<LuaLimPrefixExpr>;	// "prefixexp"
		using FUNC_CALL = LuaFuncCall;								// "functioncall"
		struct TABLE_CONSTRUCTOR { LuaTableConstructor v; };		// "tableconstructor"

		//unOps is always empty for this type
		struct MULTI_OPERATION
		{
			std::unique_ptr<struct LuaExpression> first;
			std::vector<std::pair<BinOpType, struct LuaExpression>> extra;//size>=1
		};      // "exp binop exp"

		//struct UNARY_OPERATION{UnOpType,std::unique_ptr<struct LuaExpression>};     // "unop exp"	//Inlined as opt prefix


		struct NUMERAL_I64 { int64_t v; };            // "Numeral"
	}

	using LuaExprData = std::variant<
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

	template<AnyCfgable CfgT>
	using ExprData = SelectT<CfgT, LuaExprData, LuaExprData>;

	struct LuaExpression
	{
		LuaExprData data;
		Position place;
		UnOpList unOps;

		LuaExpression() = default;
		LuaExpression(const LuaExpression&) = delete;
		LuaExpression(LuaExpression&&) = default;
		LuaExpression& operator=(LuaExpression&&) = default;
	};

	template<AnyCfgable CfgT>
	using Expression = SelectT<CfgT, LuaExpression, LuaExpression>;

	namespace SubVarType
	{
		struct NAME { std::string idx; };	// {funcArgs} ‘.’ Name

		template<bool isSlua>
		struct EXPRv { LuaExpression idx; };	// {funcArgs} ‘[’ exp ‘]’
		template<AnyCfgable CfgT> using EXPR = SelV<CfgT, EXPRv>;
	}

	template<bool isSlua>
	struct SubVarV
	{
		std::vector<LuaArgFuncCall> funcCalls;

		std::variant<
			SubVarType::NAME,
			SubVarType::EXPRv<isSlua>
		> idx;
	};

	template<AnyCfgable CfgT>
	using SubVar = SelV<CfgT, SubVarV>;

	namespace BaseVarType
	{
		using NAME = std::string;

		template<bool isSlua>
		struct EXPRv { LuaExpression start; SubVarV<isSlua> sub; };

		template<AnyCfgable CfgT> using EXPR = SelV<CfgT, EXPRv>;
	}
	template<bool isSlua>
	using BaseVarV = std::variant<
		BaseVarType::NAME,
		BaseVarType::EXPRv<isSlua>
	>;

	template<AnyCfgable CfgT>
	using BaseVar = SelV<CfgT, BaseVarV>;

	template<bool isSlua>
	struct VarV
	{
		BaseVarV<isSlua> base;
		std::vector<SubVarV<isSlua>> sub;
	};


	template<AnyCfgable CfgT>
	using Var = SelV<CfgT, VarV>;

	struct AttribName
	{
		std::string name;
		std::string attrib;//empty -> no attrib
	};

	namespace FieldType
	{
		struct EXPR2EXPR { LuaExpression idx; LuaExpression v; };		// "‘[’ exp ‘]’ ‘=’ exp"
		struct NAME2EXPR { std::string idx; LuaExpression v; };	// "Name ‘=’ exp"
		struct EXPR { LuaExpression v; };							// "exp"
	}
	namespace LimPrefixExprType
	{
		struct VAR { VarV<false> v; };			// "var"
		struct EXPR { LuaExpression v; };	// "'(' exp ')'"
	}

	using AttribNameList = std::vector<AttribName>;
	using NameList = std::vector<std::string>;

	namespace StatementType
	{
		struct SEMICOLON {};									// ";"

		template<bool isSlua>
		struct ASSIGNv { std::vector<VarV<isSlua>> vars; LuaExpList exprs; };// "varlist = explist" //e.size must be > 0
		template<AnyCfgable CfgT> using ASSIGN = SelV<CfgT, ASSIGNv>;

		template<bool isSlua>
		using FUNC_CALLv = LuaFuncCall;								// "functioncall"
		template<AnyCfgable CfgT> using FUNC_CALL = SelV<CfgT, FUNC_CALLv>;

		struct LABEL { std::string v; };						// "label"
		struct BREAK { std::string v; };						// "break"
		struct GOTO { std::string v; };							// "goto Name"

		template<bool isSlua>
		struct DO_BLOCKv { BlockV<isSlua> bl; };							// "do block end"
		template<AnyCfgable CfgT> using DO_BLOCK = SelV<CfgT, DO_BLOCKv>;

		template<bool isSlua>
		struct WHILE_LOOPv { LuaExpression cond; BlockV<isSlua> bl; };		// "while exp do block end"
		template<AnyCfgable CfgT> using WHILE_LOOP = SelV<CfgT, WHILE_LOOPv>;

		template<bool isSlua>
		struct REPEAT_UNTILv :WHILE_LOOPv<isSlua> {};						// "repeat block until exp"
		template<AnyCfgable CfgT> using REPEAT_UNTIL = SelV<CfgT, REPEAT_UNTILv>;

		// "if exp then block {elseif exp then block} [else block] end"
		template<bool isSlua>
		struct IF_THEN_ELSEv
		{
			LuaExpression cond;
			BlockV<isSlua> bl;
			std::vector<std::pair<LuaExpression, BlockV<isSlua>>> elseIfs;
			std::optional<BlockV<isSlua>> elseBlock;
		};
		template<AnyCfgable CfgT> using IF_THEN_ELSE = SelV<CfgT, IF_THEN_ELSEv>;

		// "for Name = exp , exp [, exp] do block end"
		template<bool isSlua>
		struct FOR_LOOP_NUMERICv
		{
			std::string varName;
			LuaExpression start;
			LuaExpression end;//inclusive
			std::optional<LuaExpression> step;
			BlockV<isSlua> bl;
		};
		template<AnyCfgable CfgT> using FOR_LOOP_NUMERIC = SelV<CfgT, FOR_LOOP_NUMERICv>;

		// "for namelist in explist do block end"
		template<bool isSlua>
		struct FOR_LOOP_GENERICv
		{
			NameList varNames;
			LuaExpList exprs;//size must be > 0
			BlockV<isSlua> bl;
		};
		template<AnyCfgable CfgT> using FOR_LOOP_GENERIC = SelV<CfgT, FOR_LOOP_GENERICv>;

		template<bool isSlua>
		struct FUNCTION_DEFv
		{// "function funcname funcbody"    //n may contain dots, 1 colon
			Position place; 
			std::string name; 
			LuaFunction func;
		};
		template<AnyCfgable CfgT> using FUNCTION_DEF = SelV<CfgT, FUNCTION_DEFv>;

		template<bool isSlua>
		struct LOCAL_FUNCTION_DEFv :FUNCTION_DEFv<isSlua> {};
		template<AnyCfgable CfgT> using LOCAL_FUNCTION_DEF = SelV<CfgT, LOCAL_FUNCTION_DEFv>;

		template<bool isSlua>				// "local function Name funcbody" //n may not ^^^
		struct LOCAL_ASSIGNv { AttribNameList names; LuaExpList exprs; };	// "local attnamelist [= explist]" //e.size 0 means "only define, no assign"
		template<AnyCfgable CfgT> using LOCAL_ASSIGN = SelV<CfgT, LOCAL_ASSIGNv>;
	};

	template<bool isSlua>
	using StatementDataV = std::variant<
		StatementType::SEMICOLON,		// ";"

		StatementType::ASSIGNv<isSlua>,			// "varlist = explist"
		StatementType::LOCAL_ASSIGNv<isSlua>,	// "local attnamelist [= explist]"

		StatementType::FUNC_CALLv<isSlua>,		// "functioncall"
		StatementType::LABEL,			// "label"
		StatementType::BREAK,			// "break"
		StatementType::GOTO,			// "goto Name"
		StatementType::DO_BLOCKv<isSlua>,		// "do block end"
		StatementType::WHILE_LOOPv<isSlua>,		// "while exp do block end"
		StatementType::REPEAT_UNTILv<isSlua>,	// "repeat block until exp"

		StatementType::IF_THEN_ELSEv<isSlua>,	// "if exp then block {elseif exp then block} [else block] end"

		StatementType::FOR_LOOP_NUMERICv<isSlua>,// "for Name = exp , exp [, exp] do block end"
		StatementType::FOR_LOOP_GENERICv<isSlua>,// "for namelist in explist do block end"

		StatementType::FUNCTION_DEFv<isSlua>,		// "function funcname funcbody"
		StatementType::LOCAL_FUNCTION_DEFv<isSlua>	// "local function Name funcbody"
	> ;

	template<AnyCfgable CfgT>
	using StatementData = SelV<CfgT, StatementDataV>;

	template<bool isSlua>
	struct StatementV
	{
		StatementDataV<isSlua> data;
		Position place;

		StatementV() = default;
		StatementV(const StatementV&) = delete;
		StatementV(StatementV&&) = default;
		StatementV& operator=(StatementV&&) = default;
	};
}