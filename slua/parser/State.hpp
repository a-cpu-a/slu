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
#undef CONST

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
	template<bool boxed,class T>
	struct MayBox
	{
		Sel<boxed, T, std::unique_ptr<T>> v;

		T& get() {
			if constexpr (boxed) return *v; else return v;
		}
		const T& get() const {
			if constexpr (boxed) return *v; else return v;
		}

		T& operator*() { return get(); }
		const T& operator*() const { return get(); }

		T& operator->() { return &get(); }
		const T* operator->() const { return &get(); }
	};
	template<bool boxed,class T>
	constexpr auto mayBoxFrom(T&& v)
	{
		if constexpr (boxed)
			return MayBox<true, T>(std::make_unique<T>(std::move(v)));
		else
			return MayBox<false, T>(std::move(v));
	}
	template<class T>
	constexpr MayBox<false,T> wontBox(T&& v) {
		return MayBox<false,T>(std::move(v));
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
	template<bool isSlua>
	using LimPrefixExprV = std::variant<
		LimPrefixExprType::VARv<isSlua>,
		LimPrefixExprType::EXPRv<isSlua>
	>;
	template<AnyCfgable CfgT>
	using LimPrefixExpr = SelV<CfgT, LimPrefixExprV>;

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

	namespace StatOrExprType
	{
		template<bool isSlua>
		using BLOCKv = BlockV<isSlua>;
		template<AnyCfgable CfgT> using BLOCK = SelV<CfgT, BLOCKv>;

		template<bool isSlua>
		using EXPRv = ExpressionV<isSlua>;
		template<AnyCfgable CfgT> using EXPR = SelV<CfgT, EXPRv>;
	}
	template<bool isSlua>
	using StatOrExprV = std::variant<
		StatOrExprType::BLOCKv<isSlua>,
		StatOrExprType::EXPRv<isSlua>
	>;
	template<AnyCfgable CfgT> using StatOrExpr = SelV<CfgT, StatOrExprV>;

	template<bool isSlua> using SoeOrBlockV = Sel<isSlua,BlockV<isSlua>,StatOrExprV<isSlua>>;
	template<AnyCfgable CfgT> using SoeOrBlock = SelV<CfgT, SoeOrBlockV>;

	template<bool isSlua> using SoeBoxOrBlockV = Sel<isSlua, BlockV<isSlua>, std::unique_ptr<StatOrExprV<isSlua>>>;
	template<AnyCfgable CfgT> using SoeBoxOrBlock = SelV<CfgT, SoeBoxOrBlockV>;

	namespace ArgsType
	{
		template<bool isSlua>
		struct EXPLISTv { ExpListV<isSlua> v; };			// "'(' [explist] ')'"
		template<AnyCfgable CfgT> using EXPLIST = SelV<CfgT, EXPLISTv>;

		template<bool isSlua>
		struct TABLEv { TableConstructorV<isSlua> v; };	// "tableconstructor"
		template<AnyCfgable CfgT> using TABLE = SelV<CfgT, TABLEv>;

		struct LITERAL { std::string v; Position end; };// "LiteralString"
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
	struct FuncCallV
	{
		std::unique_ptr<LimPrefixExprV<isSlua>> val;
		std::vector<ArgFuncCallV<isSlua>> argChain;
	};


	namespace ExprType
	{

		template<bool isSlua>
		using LIM_PREFIX_EXPv = std::unique_ptr<LimPrefixExprV<isSlua>>;	// "prefixexp"
		template<AnyCfgable CfgT> using LIM_PREFIX_EXP = SelV<CfgT, LIM_PREFIX_EXPv>;

		template<bool isSlua>
		using FUNC_CALLv = FuncCallV<isSlua>;								// "functioncall"
		template<AnyCfgable CfgT> using FUNC_CALL = SelV<CfgT, FUNC_CALLv>;

		struct OPEN_RANGE {};					// "..."

		struct LITERAL_STRING { std::string v; Position end;};	// "LiteralString"
		struct NUMERAL { double v; };							// "Numeral"

		struct NUMERAL_I64 { int64_t v; };            // "Numeral"

		//u64,i128,u128, for slua only
		struct NUMERAL_U64 { uint64_t v; };						// "Numeral"
		struct NUMERAL_U128 { uint64_t lo = 0; uint64_t hi = 0; }; // "Numeral"
		struct NUMERAL_I128 :NUMERAL_U128 {};					// "Numeral"

	}
	using Lifetime = std::vector<MpItmIdV<true>>;
	struct UnOpItem
	{
		Lifetime life;
		UnOpType type;
	};

	//Slua

	namespace TraitExprItemType
	{
		using LIM_PREFIX_EXP = std::unique_ptr<LimPrefixExprV<true>>;
		using FUNC_CALL = FuncCallV<true>;
	}
	using TraitExprItem = std::variant<
		TraitExprItemType::LIM_PREFIX_EXP,
		TraitExprItemType::FUNC_CALL
	>;
	struct TraitExpr
	{
		std::vector<TraitExprItem> traitCombo;
		Position place;
	};

	namespace TypeExprDataType
	{
		using ERR_INFERR = std::monostate;

		struct TRAIT_TY {};

		using LIM_PREFIX_EXP = std::unique_ptr<LimPrefixExprV<true>>;
		using FUNC_CALL = FuncCallV<true>;

		struct MULTI_OP
		{
			std::unique_ptr<TypeExpr> first;
			std::vector<std::pair<BinOpType, TypeExpr>> extra;
		};
		using TABLE_CONSTRUCTOR = TableConstructorV<true>;

		struct FN
		{
			std::unique_ptr<TypeExpr> argType;
			std::unique_ptr<TypeExpr> retType;
			OptSafety safety = OptSafety::DEFAULT;
		};

		struct DYN
		{
			TraitExpr expr;
		};
		struct IMPL
		{
			TraitExpr expr;
		};
		using SLICER = std::unique_ptr<ExpressionV<true>>;
		struct ERR
		{
			std::unique_ptr<TypeExpr> err;
		};

		using ExprType::NUMERAL_U64;
		using ExprType::NUMERAL_I64;
		using ExprType::NUMERAL_U128;
		using ExprType::NUMERAL_I128;
	}
	using TypeExprData = std::variant<
		TypeExprDataType::ERR_INFERR,
		TypeExprDataType::TRAIT_TY,

		TypeExprDataType::LIM_PREFIX_EXP,
		TypeExprDataType::FUNC_CALL,
		TypeExprDataType::MULTI_OP,
		TypeExprDataType::TABLE_CONSTRUCTOR,
		TypeExprDataType::DYN,
		TypeExprDataType::IMPL,
		TypeExprDataType::SLICER,
		TypeExprDataType::ERR,
		TypeExprDataType::FN,

		TypeExprDataType::NUMERAL_U64,
		TypeExprDataType::NUMERAL_I64,
		TypeExprDataType::NUMERAL_U128,
		TypeExprDataType::NUMERAL_I128
	>;
	struct TypeExpr
	{
		TypeExprData data;
		Position place;
		std::vector<UnOpItem> unOps;//TODO: for lua, use small op list
		bool hasMut : 1 = false;

		bool isBasicStruct() const
		{
			return !hasMut && unOps.empty()
				&& std::holds_alternative<TypeExprDataType::TABLE_CONSTRUCTOR>(data);
		}
	};

	using TypePrefix = std::vector<UnOpItem>;




	//Common

	//NOTE: has overload later!!!
	template<bool isSlua>
	struct ParameterV
	{
		MpItmIdV<isSlua> name;
	};
	template<AnyCfgable CfgT>
	using Parameter = SelV<CfgT, ParameterV>;

	template<bool isSlua>
	using ParamListV = std::vector<ParameterV<isSlua>>;
	template<AnyCfgable CfgT> using ParamList = SelV<CfgT, ParamListV>;

	template<bool isSlua>
	struct BaseFunctionV
	{
		ParamListV<isSlua> params;
		BlockV<isSlua> block;
		bool hasVarArgParam = false;// do params end with '...'
	};

	template<bool isSlua>
	struct FunctionV : BaseFunctionV<isSlua>
	{};
	template<>
	struct FunctionV<true> : BaseFunctionV<true>
	{
		std::optional<TypeExpr> retType;
		OptSafety safety = OptSafety::DEFAULT;
	};

	template<AnyCfgable CfgT>
	using Function = SelV<CfgT, FunctionV>;



	template<bool isSlua,bool boxIt>
	struct BaseIfCondV
	{
		std::vector<std::pair<ExpressionV<isSlua>, SoeOrBlockV<isSlua>>> elseIfs;
		MayBox<boxIt, ExpressionV<isSlua>> cond;
		MayBox<boxIt, SoeOrBlockV<isSlua>> bl;
		std::optional<MayBox<boxIt, SoeOrBlockV<isSlua>>> elseBlock;
	};
	template<AnyCfgable CfgT, bool boxIt> 
	using BaseIfCond = Sel<CfgT::settings()&sluaSyn, BaseIfCondV<false,boxIt>, BaseIfCondV<true,boxIt>>;

	namespace ExprType
	{
		using NIL = std::monostate;								// "nil"
		struct FALSE {};										// "false"
		struct TRUE {};											// "true"
		struct VARARGS {};										// "..."

		template<bool isSlua>
		struct FUNCTION_DEFv { FunctionV<isSlua> v; };				// "functiondef"
		template<AnyCfgable CfgT> using FUNCTION_DEF = SelV<CfgT, FUNCTION_DEFv>;

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

		template<bool isSlua>
		using IfCondV = BaseIfCondV<isSlua, true>;
		template<AnyCfgable CfgT> using IfCond = SelV<CfgT, IfCondV>;


		using LIFETIME = Lifetime;	// " '/' var" {'/' var"}
		using TYPE_EXPR = TypeExpr;
		using TRAIT_EXPR = TraitExpr;

		struct PAT_TYPE_PREFIX {};
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

		ExprType::IfCondV<isSlua>,

		ExprType::OPEN_RANGE,			// "..."

		ExprType::NUMERAL_U64,			// "Numeral"
		ExprType::NUMERAL_I128,			// "Numeral"
		ExprType::NUMERAL_U128,			// "Numeral"

		ExprType::LIFETIME,
		ExprType::TYPE_EXPR,
		ExprType::TRAIT_EXPR,

		ExprType::PAT_TYPE_PREFIX
	>;

	template<AnyCfgable CfgT>
	using ExprData = SelV<CfgT, ExprDataV>;


	template<bool isSlua>
	struct BaseExpressionV
	{
		ExprDataV<isSlua> data;
		Position place;
		std::vector<UnOpItem> unOps;

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

	//Slua


	// match patterns

	using NdPat = ExpressionV<true>;

	namespace DestrSpecType
	{
		using Spat = parse::NdPat;
		using Type = TypeExpr;
		using Prefix = TypePrefix;
	}
	using DestrSpec = std::variant<
		DestrSpecType::Spat,
		DestrSpecType::Type,
		DestrSpecType::Prefix

	>;
	namespace DestrPatType
	{
		using Any = std::monostate;
		struct Fields;
		struct List;

		struct Name;
		struct NameRestrict;
	}
	struct DestrField;
	struct ___PatHack;
	namespace DestrPatType
	{
		struct Fields
		{
			DestrSpec spec;
			bool extraFields : 1 = false;
			std::vector<DestrField> items;
			MpItmIdV<true> name;//May be empty
		};
		struct List
		{
			DestrSpec spec;
			bool extraFields : 1 = false;
			std::vector<___PatHack> items;
			MpItmIdV<true> name;//May be empty
		};

		struct Name
		{
			MpItmIdV<true> name;
			DestrSpec spec;
		};
		struct NameRestrict : Name
		{
			NdPat restriction;
		};
	}
	namespace PatType
	{
		//x or y or z
		using Simple = NdPat;

		using DestrAny = DestrPatType::Any;

		using DestrFields = DestrPatType::Fields;
		using DestrList = DestrPatType::List;

		using DestrName = DestrPatType::Name;
		using DestrNameRestrict = DestrPatType::NameRestrict;
	}
	template<typename T>
	concept AnyCompoundDestr = 
		std::same_as<std::remove_cv_t<T>, DestrPatType::Fields>
		|| std::same_as<std::remove_cv_t<T>, DestrPatType::List>;

	using Pat = std::variant<
		PatType::DestrAny,

		PatType::Simple,

		PatType::DestrFields,
		PatType::DestrList,

		PatType::DestrName,
		PatType::DestrNameRestrict
	>;
	template<>
	struct ParameterV<true>
	{
		Pat name;
	};
	struct ___PatHack : Pat {};
	struct DestrField
	{
		MpItmIdV<true> name;
		Pat pat;
	};

	//Common

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
		using IfCondV = BaseIfCondV<isSlua, false>;
		template<AnyCfgable CfgT> using IfCond = SelV<CfgT, IfCondV>;

		// "for Name = exp , exp [, exp] do block end"
		template<bool isSlua>
		struct FOR_LOOP_NUMERICv
		{
			Sel<isSlua, MpItmIdV<isSlua>, Pat> varName;
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
			Sel<isSlua, NameListV<isSlua>, Pat> varNames;
			Sel<isSlua, ExpListV<isSlua>, ExpressionV<isSlua>> exprs;//size must be > 0
			BlockV<isSlua> bl;
		};
		template<AnyCfgable CfgT> using FOR_LOOP_GENERIC = SelV<CfgT, FOR_LOOP_GENERICv>;

		template<bool isSlua>
		struct FuncDefBase
		{// "function funcname funcbody"    
			Position place;//Right before func-name
			MpItmIdV<isSlua> name; // name may contain dots, 1 colon if !isSlua
			FunctionV<isSlua> func;
		};
		template<bool isSlua>
		struct FUNCTION_DEFv : FuncDefBase<isSlua> {};
		template<>
		struct FUNCTION_DEFv<true> : FuncDefBase<true> 
		{
			ExportData exported = false;
		};

		template<AnyCfgable CfgT> using FUNCTION_DEF = SelV<CfgT, FUNCTION_DEFv>;

		template<bool isSlua>
		struct FNv : FUNCTION_DEFv<isSlua> {};
		template<AnyCfgable CfgT> using FN = SelV<CfgT, FNv>;

		template<bool isSlua>
		struct LOCAL_FUNCTION_DEFv :FUNCTION_DEFv<isSlua> {};
		template<AnyCfgable CfgT> using LOCAL_FUNCTION_DEF = SelV<CfgT, LOCAL_FUNCTION_DEFv>;
				// "local function Name funcbody" //n may not ^^^

		template<bool isSlua>
		struct LOCAL_ASSIGNv
		{	// "local attnamelist [= explist]" //e.size 0 means "only define, no assign"
			AttribNameListV<isSlua> names;
			ExpListV<isSlua> exprs;
		};
		template<>
		struct LOCAL_ASSIGNv<true>
		{	// "local attnamelist [= explist]" //e.size 0 means "only define, no assign"
			Pat names;
			ExpListV<true> exprs;
			ExportData exported = false;
		};
		template<AnyCfgable CfgT> using LOCAL_ASSIGN = SelV<CfgT, LOCAL_ASSIGNv>;

		// Slua

		template<bool isSlua>
		struct LETv : LOCAL_ASSIGNv<isSlua>	{};
		template<AnyCfgable CfgT> using LET = SelV<CfgT, LETv>;

		template<bool isSlua>
		struct CONSTv : LOCAL_ASSIGNv<isSlua>	{};
		template<AnyCfgable CfgT> using CONST = SelV<CfgT, CONSTv>;

		template<bool isSlua>
		struct StructV
		{
			ParamListV<isSlua> params;
			TypeExpr type;
			MpItmIdV<isSlua> name;
			ExportData exported = false;
		};
		template<AnyCfgable CfgT> using Struct = SelV<CfgT, StructV>;

		struct UNSAFE_LABEL {};
		struct SAFE_LABEL {};

		struct USE
		{
			MpItmIdV<true> base;//the aliased/imported thing, or modpath base
			UseVariant useVariant;
			ExportData exported=false;
		};

		template<bool isSlua>
		struct DROPv
		{
			ExpressionV<isSlua> expr;
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

	};

	template<bool isSlua>
	using StatementDataV = std::variant<
		StatementType::SEMICOLON,				// ";"

		StatementType::ASSIGNv<isSlua>,			// "varlist = explist"
		StatementType::LOCAL_ASSIGNv<isSlua>,	// "local attnamelist [= explist]"
		StatementType::LETv<isSlua>,	// "let pat [= explist]"
		StatementType::CONSTv<isSlua>,	// "const pat [= explist]"

		StatementType::FUNC_CALLv<isSlua>,		// "functioncall"
		StatementType::LABELv<isSlua>,			// "label"
		StatementType::BREAK,					// "break"
		StatementType::GOTOv<isSlua>,			// "goto Name"
		StatementType::BLOCKv<isSlua>,			// "do block end"
		StatementType::WHILE_LOOPv<isSlua>,		// "while exp do block end"
		StatementType::REPEAT_UNTILv<isSlua>,	// "repeat block until exp"

		StatementType::IfCondV<isSlua>,	// "if exp then block {elseif exp then block} [else block] end"

		StatementType::FOR_LOOP_NUMERICv<isSlua>,	// "for Name = exp , exp [, exp] do block end"
		StatementType::FOR_LOOP_GENERICv<isSlua>,	// "for namelist in explist do block end"

		StatementType::FUNCTION_DEFv<isSlua>,		// "function funcname funcbody"
		StatementType::FNv<isSlua>,					// "fn funcname funcbody"
		StatementType::LOCAL_FUNCTION_DEFv<isSlua>,	// "local function Name funcbody"

		StatementType::StructV<isSlua>,

		StatementType::UNSAFE_LABEL,	// ::: unsafe :
		StatementType::SAFE_LABEL,		// ::: safe :

		StatementType::DROPv<isSlua>,	// "drop" Name
		StatementType::USE,				// "use" ...
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