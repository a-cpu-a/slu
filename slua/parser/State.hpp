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
#include <slua/lang/BasicState.hpp>
#include "Enums.hpp"
#include "SmallEnumList.hpp"
#include "Input.hpp"

//for enums...
#undef FALSE
#undef TRUE

namespace slua::parse
{

	template<AnyCfgable CfgT, template<bool> class T>
	using SelV = T<CfgT::settings()& sluaSyn>;

	template<bool isSlua, class T,class SlT>
	using Sel = std::conditional_t<isSlua,SlT,T>;

	template<AnyCfgable Cfg, size_t TOK_SIZE, size_t TOK_SIZE2>
	consteval const auto& sel(const char(&tok)[TOK_SIZE], const char(&sluaTok)[TOK_SIZE2])
	{
		if constexpr (Cfg::settings() & sluaSyn)
			return sluaTok;
		else
			return tok;
	}

	//Mp ref
	template<AnyCfgable CfgT> using MpItmId = SelV<CfgT, lang::MpItmIdV>;



	//Forward declare

	template<bool isSlua> struct StatementV;
	template<AnyCfgable CfgT> using Statement = SelV<CfgT, StatementV>;

	template<bool isSlua> struct ExpressionV;
	template<AnyCfgable CfgT> using Expression = SelV<CfgT, ExpressionV>;

	template<bool isSlua> struct VarV;
	template<AnyCfgable CfgT> using Var = SelV<CfgT, VarV>;

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
		template<bool isSlua> struct VARv;			// "var"
		template<AnyCfgable CfgT> using VAR = SelV<CfgT, VARv>;

		template<bool isSlua> struct EXPRv;	// "'(' exp ')'"
		template<AnyCfgable CfgT> using EXPR = SelV<CfgT, EXPRv>;
	}

	template<bool isSlua> struct ArgFuncCallV;
	template<AnyCfgable CfgT> using ArgFuncCall = SelV<CfgT, ArgFuncCallV>;

	template<bool isSlua> struct FuncCallV;
	template<AnyCfgable CfgT> using FuncCall = SelV<CfgT, FuncCallV>;

	template<bool isSlua>
	using ExpListV = std::vector<ExpressionV<isSlua>>;
	template<AnyCfgable CfgT> using ExpList = SelV<CfgT, ExpListV>;

	struct TypeExpr;

	// Slua

	//Possible future optimization:
	/*
	enum class TypeId : size_t {}; //Strong alias
	*/


	using slua::lang::MpItmIdV;
	using slua::lang::ModPath;
	using slua::lang::ModPathView;
	using slua::lang::ExportData;
	using SubModPath = std::vector<std::string>;

	struct BorrowLevel
	{
		std::vector<MpItmIdV<true>> lifetimes;
		bool hasMut=false;//eventualy share too
	};
	struct GcPtrLevel
	{
		bool isPtr : 1 = true;// false -> gc
		bool hasMut : 1 = false;
	};
	struct TypeSpecifiers
	{
		size_t derefCount = 0;
		std::vector<BorrowLevel> borrows;
		std::vector<GcPtrLevel> gcPtrLevels;
	};

	using TraitCombo = std::vector<ModPath>;
	using Type = std::vector<struct TypeItem>;
	
	struct TupleItem
	{
		Type ty;
		MpItmIdV<true> name;// empty -> no name
		bool hasMut : 1 = false;
	};

	namespace TypeObjType
	{
		using COMPTIME_VAR_TYPE = ModPath;
		struct SLICE_OR_ARRAY
		{
			Type ty;
			ExpListV<true> size;//empty -> slice, else array
		};
		using TUPLE = std::vector<TupleItem>;
		struct TRAIT_COMBO
		{
			TraitCombo traits;
			bool isDyn = false;
		};
		using TYPE = Type;				//"(" type ")"
	}
	using TypeObj = std::variant<
		TypeObjType::COMPTIME_VAR_TYPE,
		TypeObjType::SLICE_OR_ARRAY,
		TypeObjType::TUPLE,
		TypeObjType::TRAIT_COMBO,
		TypeObjType::TYPE>;
	struct TypeItem
	{
		TypeSpecifiers prefix;
		TypeObj obj;

		TypeItem() = default;
		TypeItem(const TypeItem&) = delete;
		TypeItem(TypeItem&&) = default;
		TypeItem& operator=(TypeItem&&) = default;
	};


	namespace BasicTypeObjType
	{
		using COMPTIME_VAR_TYPE = TypeObjType::COMPTIME_VAR_TYPE;
		using TRAIT_COMBO = TypeObjType::TRAIT_COMBO;
	}
	using BasicTypeObj = std::variant<
		BasicTypeObjType::COMPTIME_VAR_TYPE,
		BasicTypeObjType::TRAIT_COMBO>;
	struct BasicTypeItem
	{
		TypeSpecifiers prefix;
		BasicTypeObj obj;
	};
	using BasicType = std::vector<BasicTypeItem>;
	using BasicTypeOrPrefix = std::variant<BasicType,TypeSpecifiers>;

	struct ErrType
	{
		std::optional<Type> val;
		std::optional<Type> err;//If missing then ??, else ?
		bool isErr = false;
	};

	// Common


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
		MpItmIdV<isSlua> name;
	};

	template<>
	struct ParameterV<true> : ParameterV<false>
	{
		Type ty;
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
		ErrType retType;
		OptSafety safety = OptSafety::DEFAULT;
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

		MpItmIdV<isSlua> funcName;//If empty, then no colon needed. Only used for ":xxx"
		ArgsV<isSlua> args;
	};


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



	//Slua

	namespace TraitExprItemType
	{
		using EXPR = struct ExpressionV<true>;
		using FUNC_CALL = struct FuncCallV<true>;
		using VAR = VarV<true>;
	}
	using TraitExprItem = std::variant<
		TraitExprItemType::EXPR,
		TraitExprItemType::FUNC_CALL,
		TraitExprItemType::VAR
	>;
	struct TraitExpr
	{
		std::vector<TraitExprItem> traitCombo;
		Position place;
	};

	namespace TypeExprDataType
	{
		using ERR_INFERR = std::monostate;

		using EXPR = std::unique_ptr<struct ExpressionV<true>>;
		using FUNC_CALL = struct FuncCallV<true>;
		using VAR = std::unique_ptr<struct VarV<true>>;
		struct MULTI_OP
		{
			std::unique_ptr<TypeExpr> first;
			std::vector<std::pair<BinOpType, TypeExpr>> extra;
		};
		using TABLE_CONSTRUCTOR = TableConstructorV<true>;

		struct DYN
		{
			TraitExpr expr;
		};
		struct IMPL
		{
			TraitExpr expr;
		};
		using SLICER = std::unique_ptr<TypeExpr>;
		struct ERR
		{
			std::unique_ptr<TypeExpr> err;
		};
	}
	using TypeExprData = std::variant<
		TypeExprDataType::ERR_INFERR,
		TypeExprDataType::EXPR,
		TypeExprDataType::FUNC_CALL,
		TypeExprDataType::VAR,
		TypeExprDataType::MULTI_OP,
		TypeExprDataType::TABLE_CONSTRUCTOR,
		TypeExprDataType::DYN,
		TypeExprDataType::IMPL,
		TypeExprDataType::SLICER,
		TypeExprDataType::ERR
	>;
	struct TypeExpr
	{
		TypeExprData data;
		Position place;
		SmallEnumList<UnOpType> unOps;
		bool hasMut : 1 = false;
	};
	//Common

	namespace ExprType
	{
		using NIL = std::monostate;								// "nil"
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
		struct NUMERAL_U64 { uint64_t v; };						// "Numeral"
		struct NUMERAL_U128 { uint64_t lo=0; uint64_t hi = 0;}; // "Numeral"
		struct NUMERAL_I128 :NUMERAL_U128 {};					// "Numeral"

		struct OPEN_RANGE { };					// "..."

		using LIFETIME = std::vector<MpItmIdV<true>>;	// " '/' var" {'/' var"}
		using TYPE_EXPR = TypeExpr;
		using TRAIT_EXPR = TraitExpr;

		template<bool isSlua>
		struct ARRAY_CONSTRUCTORv
		{
			std::unique_ptr<ExpressionV<isSlua>> val;
			std::unique_ptr<ExpressionV<isSlua>> size;
		};
		template<AnyCfgable CfgT> 
		using ARRAY_CONSTRUCTOR = SelV<CfgT, ARRAY_CONSTRUCTORv>;
	}

	template<bool isSlua>
	using ExprDataV = std::variant<
		ExprType::NIL,                  // "nil"
		ExprType::FALSE,                // "false"
		ExprType::TRUE,                 // "true"
		ExprType::NUMERAL,				// "Numeral" (e.g., a floating-point number)
		ExprType::NUMERAL_I64,			// "Numeral"

		ExprType::LITERAL_STRING,		// "LiteralString"
		ExprType::VARARGS,              // "..." (varargs)
		ExprType::FUNCTION_DEFv<isSlua>,			// "functiondef"
		ExprType::LIM_PREFIX_EXPv<isSlua>,		// "prefixexp"
		ExprType::FUNC_CALLv<isSlua>,			// "prefixexp argsThing {argsThing}"
		ExprType::TABLE_CONSTRUCTORv<isSlua>,	// "tableconstructor"

		ExprType::MULTI_OPERATIONv<isSlua>,		// "exp binop exp {binop exp}"  // added {binop exp}, cuz multi-op

		// Slua

		ExprType::OPEN_RANGE,			// "..."

		ExprType::NUMERAL_U64,			// "Numeral"
		ExprType::NUMERAL_I128,			// "Numeral"
		ExprType::NUMERAL_U128,			// "Numeral"

		ExprType::LIFETIME,
		ExprType::TYPE_EXPR,
		ExprType::TRAIT_EXPR,

		ExprType::ARRAY_CONSTRUCTORv<isSlua>
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
		SmallEnumList<PostUnOpType> postUnOps;
	};

	namespace SubVarType
	{
		template<bool isSlua>
		struct NAMEv { MpItmIdV<isSlua> idx; };	// {funcArgs} ‘.’ Name
		template<AnyCfgable CfgT> using NAME = SelV<CfgT, NAMEv>;

		template<bool isSlua>
		struct EXPRv { ExpressionV<isSlua> idx; };	// {funcArgs} ‘[’ exp ‘]’
		template<AnyCfgable CfgT> using EXPR = SelV<CfgT, EXPRv>;
	}

	template<bool isSlua>
	struct SubVarV
	{
		std::vector<ArgFuncCallV<isSlua>> funcCalls;

		std::variant<
			SubVarType::NAMEv<isSlua>,
			SubVarType::EXPRv<isSlua>
		> idx;
	};

	template<AnyCfgable CfgT>
	using SubVar = SelV<CfgT, SubVarV>;

	namespace BaseVarType
	{
		template<bool isSlua>
		struct NAMEv
		{
			MpItmIdV<isSlua> v;
		};
		template<>
		struct NAMEv<true>
		{
			MpItmIdV<true> v;
			bool hasDeref=false;
		};
		template<AnyCfgable CfgT>
		using NAME = SelV<CfgT, NAMEv>;

		template<bool isSlua>
		struct EXPRv
		{
			ExpressionV<isSlua> start;
		};
		template<AnyCfgable CfgT> using EXPR = SelV<CfgT, EXPRv>;

		// Slua only:
		

		template<bool isSlua>
		struct EXPR_DEREF_NO_SUBv
		{
			ExpressionV<isSlua> start;
		};
		template<AnyCfgable CfgT> using EXPR_DEREF_NO_SUB = SelV<CfgT, EXPR_DEREF_NO_SUBv>;

	}
	template<bool isSlua>
	using BaseVarV = std::variant<
		BaseVarType::NAMEv<isSlua>,
		BaseVarType::EXPRv<isSlua>,
		BaseVarType::EXPR_DEREF_NO_SUBv<isSlua>
	>;

	template<AnyCfgable CfgT>
	using BaseVar = SelV<CfgT, BaseVarV>;

	template<bool isSlua>
	struct VarV
	{
		BaseVarV<isSlua> base;
		std::vector<SubVarV<isSlua>> sub;
	};

	template<bool isSlua>
	struct AttribNameV
	{
		MpItmIdV<isSlua> name;
		std::string attrib;//empty -> no attrib
	};
	template<AnyCfgable CfgT> using AttribName = SelV<CfgT, AttribNameV>;

	namespace FieldType
	{
		template<bool isSlua>
		struct EXPR2EXPRv { ExpressionV<isSlua> idx; ExpressionV<isSlua> v; };		// "‘[’ exp ‘]’ ‘=’ exp"

		template<bool isSlua>
		struct NAME2EXPRv { MpItmIdV<isSlua> idx; ExpressionV<isSlua> v; };	// "Name ‘=’ exp"

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

	template<bool isSlua>
	using AttribNameListV = std::vector<AttribNameV<isSlua>>;
	template<AnyCfgable CfgT> using AttribNameList = SelV<CfgT, AttribNameListV>;
	template<bool isSlua>
	using NameListV = std::vector<MpItmIdV<isSlua>>;
	template<AnyCfgable CfgT> using NameList = SelV<CfgT, NameListV>;

	namespace UseVariantType
	{
		using EVERYTHING_INSIDE = std::monostate;//use x::*;
		struct IMPORT {};// use x::y;
		using AS_NAME = MpItmIdV<true>;//use x as y;
		using LIST_OF_STUFF = std::vector<MpItmIdV<true>>;//use x::{self, ...}
	}
	using UseVariant = std::variant<
		UseVariantType::EVERYTHING_INSIDE,
		UseVariantType::AS_NAME,
		UseVariantType::IMPORT,
		UseVariantType::LIST_OF_STUFF
	>;

	namespace StatementType
	{
		using SEMICOLON = std::monostate;	// ";"

		template<bool isSlua>
		struct ASSIGNv { std::vector<VarV<isSlua>> vars; ExpListV<isSlua> exprs; };// "varlist = explist" //e.size must be > 0
		template<AnyCfgable CfgT> using ASSIGN = SelV<CfgT, ASSIGNv>;

		template<bool isSlua>
		using FUNC_CALLv = FuncCallV<isSlua>;								// "functioncall"
		template<AnyCfgable CfgT> using FUNC_CALL = SelV<CfgT, FUNC_CALLv>;

		template<bool isSlua>
		struct LABELv { MpItmIdV<isSlua> v; };		// "label"
		template<AnyCfgable CfgT> using LABEL = SelV<CfgT, LABELv>;
		struct BREAK { };
		template<bool isSlua>					// "break"
		struct GOTOv { MpItmIdV<isSlua> v; };			// "goto Name"
		template<AnyCfgable CfgT> using GOTO = SelV<CfgT, GOTOv>;

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
			MpItmIdV<isSlua> varName;
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
			NameListV<isSlua> varNames;
			ExpListV<isSlua> exprs;//size must be > 0
			BlockV<isSlua> bl;
		};
		template<AnyCfgable CfgT> using FOR_LOOP_GENERIC = SelV<CfgT, FOR_LOOP_GENERICv>;

		template<bool isSlua>
		struct FUNCTION_DEFv
		{// "function funcname funcbody"    //n may contain dots, 1 colon
			Position place; 
			MpItmIdV<isSlua> name;
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
			Sel<isSlua, AttribNameListV<isSlua>, NameListV<isSlua>> names;
			ExpListV<isSlua> exprs;
		};
		template<AnyCfgable CfgT> using LOCAL_ASSIGN = SelV<CfgT, LOCAL_ASSIGNv>;


		// Slua

		struct UNSAFE_LABEL {};
		struct SAFE_LABEL {};

		struct USE
		{
			ModPath base;//the aliased/imported thing, or modpath base
			UseVariant useVariant;
			ExportData exported=false;
		};

		template<bool isSlua>
		struct DROPv
		{
			MpItmIdV<isSlua> var;
		};
		template<AnyCfgable CfgT> using DROP = SelV<CfgT, DROPv>;

		template<bool isSlua>
		struct MOD_DEFv
		{
			MpItmIdV<isSlua> name;
			ExportData exported = false;
		};
		template<AnyCfgable CfgT> using MOD_DEF = SelV<CfgT, MOD_DEFv>;

		template<bool isSlua>
		struct MOD_DEF_INLINEv
		{ 
			MpItmIdV<isSlua> name;
			BlockV<isSlua> bl;
			ExportData exported = false;
		};
		template<AnyCfgable CfgT> using MOD_DEF_INLINE = SelV<CfgT, MOD_DEF_INLINEv>;

		struct MOD_SELF {};
		struct MOD_CRATE {};
	};

	template<bool isSlua>
	using StatementDataV = std::variant<
		StatementType::SEMICOLON,				// ";"

		StatementType::ASSIGNv<isSlua>,			// "varlist = explist"
		StatementType::LOCAL_ASSIGNv<isSlua>,	// "local attnamelist [= explist]"

		StatementType::FUNC_CALLv<isSlua>,		// "functioncall"
		StatementType::LABELv<isSlua>,			// "label"
		StatementType::BREAK,					// "break"
		StatementType::GOTOv<isSlua>,			// "goto Name"
		StatementType::BLOCKv<isSlua>,			// "do block end"
		StatementType::WHILE_LOOPv<isSlua>,		// "while exp do block end"
		StatementType::REPEAT_UNTILv<isSlua>,	// "repeat block until exp"

		StatementType::IF_THEN_ELSEv<isSlua>,	// "if exp then block {elseif exp then block} [else block] end"

		StatementType::FOR_LOOP_NUMERICv<isSlua>,	// "for Name = exp , exp [, exp] do block end"
		StatementType::FOR_LOOP_GENERICv<isSlua>,	// "for namelist in explist do block end"

		StatementType::FUNCTION_DEFv<isSlua>,		// "function funcname funcbody"
		StatementType::LOCAL_FUNCTION_DEFv<isSlua>,	// "local function Name funcbody"

		StatementType::UNSAFE_LABEL,	// ::: unsafe :
		StatementType::SAFE_LABEL,		// ::: safe :

		StatementType::DROPv<isSlua>,	// "drop" Name
		StatementType::USE,				// "use" ...
		StatementType::MOD_SELF,				// "mod" "self"
		StatementType::MOD_CRATE,				// "mod" "crate"
		StatementType::MOD_DEFv<isSlua>,		// "mod" Name
		StatementType::MOD_DEF_INLINEv<isSlua>	// "mod" Name "as" "{" block "}"
	> ;

	template<AnyCfgable CfgT>
	using StatementData = SelV<CfgT, StatementDataV>;

	template<bool isSlua>
	struct StatementV
	{
		StatementDataV<isSlua> data;
		Position place;

		StatementV() = default;
		StatementV(StatementDataV<isSlua>&& data) :data(std::move(data)) {}
		StatementV(const StatementV&) = delete;
		StatementV(StatementV&&) = default;
		StatementV& operator=(StatementV&&) = default;
	};


	template<bool isSlua>
	struct ParsedFileV
	{
		//TypeList types
		BlockV<isSlua> code;
	};
	template<AnyCfgable CfgT>
	using ParsedFile = SelV<CfgT, ParsedFileV>;
}