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

#include <slua/ext/ExtendVariant.hpp>
#include "Enums.hpp"
#include "SmallEnumList.hpp"
#include "Input.hpp"

//for enums...
#undef FALSE
#undef TRUE

namespace sluaParse
{
	template<AnyCfgable CfgT, template<bool> class T>
	using SelV = T<CfgT::settings()& sluaSyn>;

	template<bool isSlua, class T,class SlT>
	using Sel = std::conditional_t<isSlua,SlT,T>;



	//Forward declare

	template<bool isSlua> struct StatementV;
	template<AnyCfgable CfgT> using Statement = SelV<CfgT, StatementV>;

	template<bool isSlua> struct ExpressionV;
	template<AnyCfgable CfgT> using Expression = SelV<CfgT, ExpressionV>;

	namespace FieldType {
		template<bool isSlua> struct EXPR2EXPRv;
		template<AnyCfgable CfgT> using EXPR2EXPR = SelV<CfgT, EXPR2EXPRv>;

		template<bool isSlua> struct NAME2EXPRv;
		template<AnyCfgable CfgT> using NAME2EXPR = SelV<CfgT, NAME2EXPRv>;

		template<bool isSlua> struct EXPRv;
		template<AnyCfgable CfgT> using EXPR = SelV<CfgT, EXPRv>;
	}
	namespace LimPrefixExprType
	{
		template<bool isSlua>
		struct VARv;			// "var"
		template<AnyCfgable CfgT> using VAR = SelV<CfgT, VARv>;

		template<bool isSlua>
		struct EXPRv;	// "'(' exp ')'"
		template<AnyCfgable CfgT> using EXPR = SelV<CfgT, EXPRv>;
	}





	struct TypeId
	{
		// ???
		// 32 bytes+
		// OR, 8/4 bytes for index into big array.
	};



	namespace FieldType { struct NONE {}; }

	template<bool isSlua>
	using FieldV = std::variant<
		FieldType::NONE,// Here, so variant has a default value (DO NOT USE)

		FieldType::EXPR2EXPRv<isSlua>, // "'[' exp ']' = exp"
		FieldType::NAME2EXPRv<isSlua>, // "Name = exp"
		FieldType::EXPRv<isSlua>       // "exp"
	>;

	template<AnyCfgable CfgT>
	using Field = SelV<CfgT, FieldV>;

	// ‘{’ [fieldlist] ‘}’
	template<bool isSlua>
	using TableConstructorV = std::vector<FieldV<isSlua>>;

	template<AnyCfgable CfgT>
	using TableConstructor = SelV<CfgT, TableConstructorV>;



	template<bool isSlua>
	using ExpListV = std::vector<ExpressionV<isSlua>>;

	template<AnyCfgable CfgT>
	using ExpList = SelV<CfgT, ExpListV>;

	template<bool isSlua>
	struct BlockV
	{
		std::vector<StatementV<isSlua>> statList;
		ExpListV<isSlua> retExprs;//Special, may contain 0 elements (even with hadReturn)

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


	template<bool isSlua>
	struct ParameterV
	{
		std::string name;
	};

	template<>
	struct ParameterV<true> : ParameterV<false>
	{
		TypeId ty;
	};

	template<AnyCfgable CfgT>
	using Parameter = SelV<CfgT, ParameterV>;

	template<bool isSlua>
	struct BaseFunctionV
	{
		std::vector<ParameterV<isSlua>> params;
		BlockV<isSlua> block;
		bool hasVarArgParam = false;// do params end with '...'
	};

	template<bool isSlua>
	struct FunctionV : BaseFunctionV<isSlua>
	{
	};
	template<>
	struct FunctionV<true> : BaseFunctionV<true>
	{
		TypeId retType;
	};

	template<AnyCfgable CfgT>
	using Function = SelV<CfgT, FunctionV>;

	namespace ArgsType
	{
		template<bool isSlua>
		struct EXPLISTv { ExpListV<isSlua> v; };			// "'(' [explist] ')'"
		template<AnyCfgable CfgT> using EXPLIST = SelV<CfgT, EXPLISTv>;

		template<bool isSlua>
		struct TABLEv { TableConstructorV<isSlua> v; };	// "tableconstructor"
		template<AnyCfgable CfgT> using TABLE = SelV<CfgT, TABLEv>;

		struct LITERAL { std::string v; };		// "LiteralString"
	};
	template<bool isSlua>
	using ArgsV = std::variant<
		ArgsType::EXPLISTv<isSlua>,
		ArgsType::TABLEv<isSlua>,
		ArgsType::LITERAL
	>;

	template<AnyCfgable CfgT>
	using Args = SelV<CfgT, ArgsV>;

	template<bool isSlua>
	struct ArgFuncCallV
	{// funcArgs ::=  [‘:’ Name] args

		std::string funcName;//If empty, then no colon needed. Only used for ":xxx"
		ArgsV<isSlua> args;
	};

	template<AnyCfgable CfgT>
	using ArgFuncCall = SelV<CfgT, ArgFuncCallV>;


	template<bool isSlua>
	using LimPrefixExprV = std::variant<
		LimPrefixExprType::VARv<isSlua>,
		LimPrefixExprType::EXPRv<isSlua>
	>;

	template<AnyCfgable CfgT>
	using LimPrefixExpr = SelV<CfgT, LimPrefixExprV>;

	template<bool isSlua>
	struct FuncCallV
	{
		std::unique_ptr<LimPrefixExprV<isSlua>> val;
		std::vector<ArgFuncCallV<isSlua>> argChain;
	};

	template<AnyCfgable CfgT>
	using FuncCall = SelV<CfgT, FuncCallV>;


	namespace ExprType
	{
		struct NIL {};											// "nil"
		struct FALSE {};										// "false"
		struct TRUE {};											// "true"
		struct NUMERAL { double v; };							// "Numeral"
		struct LITERAL_STRING { std::string v; };				// "LiteralString"
		struct VARARGS {};										// "..."

		template<bool isSlua>
		struct FUNCTION_DEFv { FunctionV<isSlua> v; };				// "functiondef"
		template<AnyCfgable CfgT> using FUNCTION_DEF = SelV<CfgT, FUNCTION_DEFv>;

		template<bool isSlua>
		using LIM_PREFIX_EXPv = std::unique_ptr<LimPrefixExprV<isSlua>>;	// "prefixexp"
		template<AnyCfgable CfgT> using LIM_PREFIX_EXP = SelV<CfgT, LIM_PREFIX_EXPv>;

		template<bool isSlua>
		using FUNC_CALLv = FuncCallV<isSlua>;								// "functioncall"
		template<AnyCfgable CfgT> using FUNC_CALL = SelV<CfgT, FUNC_CALLv>;

		template<bool isSlua>
		struct TABLE_CONSTRUCTORv { TableConstructorV<isSlua> v; };	// "tableconstructor"
		template<AnyCfgable CfgT> using TABLE_CONSTRUCTOR = SelV<CfgT, TABLE_CONSTRUCTORv>;

		//unOps is always empty for this type
		template<bool isSlua>
		struct MULTI_OPERATIONv
		{
			std::unique_ptr<ExpressionV<isSlua>> first;
			std::vector<std::pair<BinOpType, ExpressionV<isSlua>>> extra;//size>=1
		};      // "exp binop exp"
		template<AnyCfgable CfgT> using MULTI_OPERATION = SelV<CfgT, MULTI_OPERATIONv>;

		//struct UNARY_OPERATION{UnOpType,std::unique_ptr<ExpressionV<isSlua>>};     // "unop exp"	//Inlined as opt prefix

		struct NUMERAL_I64 { int64_t v; };            // "Numeral"

		//u64,i128,u128, for slua only
		struct NUMERAL_U64 { uint64_t v; };					// "Numeral"
		struct NUMERAL_U128 { uint64_t lo=0; uint64_t hi = 0;};   // "Numeral"
		struct NUMERAL_I128 :NUMERAL_U128 {};   // "Numeral"
	}

	template<bool isSlua>
	using ExprDataV = std::variant<
		ExprType::NIL,                  // "nil"
		ExprType::FALSE,                // "false"
		ExprType::TRUE,                 // "true"
		ExprType::NUMERAL,				// "Numeral" (e.g., a floating-point number)
		ExprType::NUMERAL_I64,			// "Numeral"

		ExprType::NUMERAL_U64,			// "Numeral"
		ExprType::NUMERAL_I128,			// "Numeral"
		ExprType::NUMERAL_U128,			// "Numeral"

		ExprType::LITERAL_STRING,		// "LiteralString"
		ExprType::VARARGS,              // "..." (varargs)
		ExprType::FUNCTION_DEFv<isSlua>,			// "functiondef"
		ExprType::LIM_PREFIX_EXPv<isSlua>,		// "prefixexp"
		ExprType::FUNC_CALLv<isSlua>,			// "prefixexp argsThing {argsThing}"
		ExprType::TABLE_CONSTRUCTORv<isSlua>,	// "tableconstructor"

		ExprType::MULTI_OPERATIONv<isSlua>		// "exp binop exp {binop exp}"  // added {binop exp}, cuz multi-op

		//ExprType::UNARY_OPERATION,	// "unop exp"
	>;

	template<AnyCfgable CfgT>
	using ExprData = SelV<CfgT, ExprDataV>;


	template<bool isSlua>
	struct BaseExpressionV
	{
		ExprDataV<isSlua> data;
		Position place;
		SmallEnumList<UnOpType> unOps;

		BaseExpressionV() = default;
		BaseExpressionV(const BaseExpressionV&) = delete;
		BaseExpressionV(BaseExpressionV&&) = default;
		BaseExpressionV& operator=(BaseExpressionV&&) = default;
	};

	template<bool isSlua>
	struct ExpressionV : BaseExpressionV<isSlua>
	{
	};
	template<>
	struct ExpressionV<true> : BaseExpressionV<true>
	{
		SmallEnumList<UnOpType> postUnOps;
	};

	namespace SubVarType
	{
		struct NAME { std::string idx; };	// {funcArgs} ‘.’ Name

		template<bool isSlua>
		struct EXPRv { ExpressionV<isSlua> idx; };	// {funcArgs} ‘[’ exp ‘]’
		template<AnyCfgable CfgT> using EXPR = SelV<CfgT, EXPRv>;
	}

	template<bool isSlua>
	struct SubVarV
	{
		std::vector<ArgFuncCallV<isSlua>> funcCalls;

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
		struct EXPRv { ExpressionV<isSlua> start; SubVarV<isSlua> sub; };

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
		template<bool isSlua>
		struct EXPR2EXPRv { ExpressionV<isSlua> idx; ExpressionV<isSlua> v; };		// "‘[’ exp ‘]’ ‘=’ exp"

		template<bool isSlua>
		struct NAME2EXPRv { std::string idx; ExpressionV<isSlua> v; };	// "Name ‘=’ exp"

		template<bool isSlua>
		struct EXPRv { ExpressionV<isSlua> v; };							// "exp"
	}
	namespace LimPrefixExprType
	{
		template<bool isSlua>
		struct VARv { VarV<isSlua> v; };			// "var"

		template<bool isSlua>
		struct EXPRv { ExpressionV<isSlua> v; };	// "'(' exp ')'"
	}

	using AttribNameList = std::vector<AttribName>;
	using NameList = std::vector<std::string>;

	namespace StatementType
	{
		struct SEMICOLON {};									// ";"

		template<bool isSlua>
		struct ASSIGNv { std::vector<VarV<isSlua>> vars; ExpListV<isSlua> exprs; };// "varlist = explist" //e.size must be > 0
		template<AnyCfgable CfgT> using ASSIGN = SelV<CfgT, ASSIGNv>;

		template<bool isSlua>
		using FUNC_CALLv = FuncCallV<isSlua>;								// "functioncall"
		template<AnyCfgable CfgT> using FUNC_CALL = SelV<CfgT, FUNC_CALLv>;

		struct LABEL { std::string v; };						// "label"
		struct BREAK { std::string v; };						// "break"
		struct GOTO { std::string v; };							// "goto Name"

		template<bool isSlua>
		struct BLOCKv { BlockV<isSlua> bl; };							// "do block end"
		template<AnyCfgable CfgT> using BLOCK = SelV<CfgT, BLOCKv>;

		template<bool isSlua>
		struct WHILE_LOOPv { ExpressionV<isSlua> cond; BlockV<isSlua> bl; };		// "while exp do block end"
		template<AnyCfgable CfgT> using WHILE_LOOP = SelV<CfgT, WHILE_LOOPv>;

		template<bool isSlua>
		struct REPEAT_UNTILv :WHILE_LOOPv<isSlua> {};						// "repeat block until exp"
		template<AnyCfgable CfgT> using REPEAT_UNTIL = SelV<CfgT, REPEAT_UNTILv>;

		// "if exp then block {elseif exp then block} [else block] end"
		template<bool isSlua>
		struct IF_THEN_ELSEv
		{
			ExpressionV<isSlua> cond;
			BlockV<isSlua> bl;
			std::vector<std::pair<ExpressionV<isSlua>, BlockV<isSlua>>> elseIfs;
			std::optional<BlockV<isSlua>> elseBlock;
		};
		template<AnyCfgable CfgT> using IF_THEN_ELSE = SelV<CfgT, IF_THEN_ELSEv>;

		// "for Name = exp , exp [, exp] do block end"
		template<bool isSlua>
		struct FOR_LOOP_NUMERICv
		{
			std::string varName;
			ExpressionV<isSlua> start;
			ExpressionV<isSlua> end;//inclusive
			std::optional<ExpressionV<isSlua>> step;
			BlockV<isSlua> bl;
		};
		template<AnyCfgable CfgT> using FOR_LOOP_NUMERIC = SelV<CfgT, FOR_LOOP_NUMERICv>;

		// "for namelist in explist do block end"
		template<bool isSlua>
		struct FOR_LOOP_GENERICv
		{
			NameList varNames;
			ExpListV<isSlua> exprs;//size must be > 0
			BlockV<isSlua> bl;
		};
		template<AnyCfgable CfgT> using FOR_LOOP_GENERIC = SelV<CfgT, FOR_LOOP_GENERICv>;

		template<bool isSlua>
		struct FUNCTION_DEFv
		{// "function funcname funcbody"    //n may contain dots, 1 colon
			Position place; 
			std::string name; 
			FunctionV<isSlua> func;
		};
		template<AnyCfgable CfgT> using FUNCTION_DEF = SelV<CfgT, FUNCTION_DEFv>;

		template<bool isSlua>
		struct LOCAL_FUNCTION_DEFv :FUNCTION_DEFv<isSlua> {};
		template<AnyCfgable CfgT> using LOCAL_FUNCTION_DEF = SelV<CfgT, LOCAL_FUNCTION_DEFv>;
				// "local function Name funcbody" //n may not ^^^

		template<bool isSlua>
		struct LOCAL_ASSIGNv
		{	// "local attnamelist [= explist]" //e.size 0 means "only define, no assign"
			Sel<isSlua, AttribNameList, NameList> names;
			ExpListV<isSlua> exprs;
		};
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
		StatementType::BLOCKv<isSlua>,		// "do block end"
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