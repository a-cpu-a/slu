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

#include <slu/ext/ExtendVariant.hpp>
#include <slu/lang/BasicState.hpp>
#include "Enums.hpp"
#include "SmallEnumList.hpp"
#include "Input.hpp"

//for enums...
#undef FALSE
#undef TRUE
#undef CONST

namespace slu::parse
{

	template<AnyCfgable CfgT, template<bool> class T>
	using SelV = T<CfgT::settings()& sluSyn>;

	template<bool isSlu, class T,class SlT>
	using Sel = std::conditional_t<isSlu,SlT,T>;

	template<AnyCfgable Cfg, size_t TOK_SIZE, size_t TOK_SIZE2>
	consteval const auto& sel(const char(&tok)[TOK_SIZE], const char(&sluTok)[TOK_SIZE2])
	{
		if constexpr (Cfg::settings() & sluSyn)
			return sluTok;
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

	template<bool isSlu> struct StatementV;
	template<AnyCfgable CfgT> using Statement = SelV<CfgT, StatementV>;

	template<bool isSlu> struct ExpressionV;
	template<AnyCfgable CfgT> using Expression = SelV<CfgT, ExpressionV>;

	template<bool isSlu> struct VarV;
	template<AnyCfgable CfgT> using Var = SelV<CfgT, VarV>;

	namespace FieldType {
		template<bool isSlu> struct EXPR2EXPRv;
		template<AnyCfgable CfgT> using EXPR2EXPR = SelV<CfgT, EXPR2EXPRv>;

		template<bool isSlu> struct NAME2EXPRv;
		template<AnyCfgable CfgT> using NAME2EXPR = SelV<CfgT, NAME2EXPRv>;

		template<bool isSlu> struct EXPRv;
		template<AnyCfgable CfgT> using EXPR = SelV<CfgT, EXPRv>;
	}
	namespace LimPrefixExprType
	{
		template<bool isSlu> struct VARv;			// "var"
		template<AnyCfgable CfgT> using VAR = SelV<CfgT, VARv>;

		template<bool isSlu> struct EXPRv;	// "'(' exp ')'"
		template<AnyCfgable CfgT> using EXPR = SelV<CfgT, EXPRv>;
	}
	template<bool isSlu>
	using LimPrefixExprV = std::variant<
		LimPrefixExprType::VARv<isSlu>,
		LimPrefixExprType::EXPRv<isSlu>
	>;
	template<AnyCfgable CfgT>
	using LimPrefixExpr = SelV<CfgT, LimPrefixExprV>;

	template<bool isSlu> struct ArgFuncCallV;
	template<AnyCfgable CfgT> using ArgFuncCall = SelV<CfgT, ArgFuncCallV>;

	template<bool isSlu> struct FuncCallV;
	template<AnyCfgable CfgT> using FuncCall = SelV<CfgT, FuncCallV>;

	template<bool isSlu>
	using ExpListV = std::vector<ExpressionV<isSlu>>;
	template<AnyCfgable CfgT> using ExpList = SelV<CfgT, ExpListV>;

	struct TypeExpr;

	// Slu

	//Possible future optimization:
	/*
	enum class TypeId : size_t {}; //Strong alias
	*/


	using slu::lang::MpItmIdV;
	using slu::lang::ModPath;
	using slu::lang::ModPathView;
	using slu::lang::ExportData;
	using SubModPath = std::vector<std::string>;


	// Common


	namespace FieldType { struct NONE {}; }

	template<bool isSlu>
	using FieldV = std::variant<
		FieldType::NONE,// Here, so variant has a default value (DO NOT USE)

		FieldType::EXPR2EXPRv<isSlu>, // "'[' exp ']' = exp"
		FieldType::NAME2EXPRv<isSlu>, // "Name = exp"
		FieldType::EXPRv<isSlu>       // "exp"
	>;

	template<AnyCfgable CfgT>
	using Field = SelV<CfgT, FieldV>;

	// ‘{’ [fieldlist] ‘}’
	template<bool isSlu>
	using TableConstructorV = std::vector<FieldV<isSlu>>;
	template<AnyCfgable CfgT>
	using TableConstructor = SelV<CfgT, TableConstructorV>;




	template<bool isSlu>
	struct BlockV
	{
		std::vector<StatementV<isSlu>> statList;
		ExpListV<isSlu> retExprs;//Special, may contain 0 elements (even with hadReturn)

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
		template<bool isSlu>
		using BLOCKv = BlockV<isSlu>;
		template<AnyCfgable CfgT> using BLOCK = SelV<CfgT, BLOCKv>;

		template<bool isSlu>
		using EXPRv = ExpressionV<isSlu>;
		template<AnyCfgable CfgT> using EXPR = SelV<CfgT, EXPRv>;
	}
	template<bool isSlu>
	using StatOrExprV = std::variant<
		StatOrExprType::BLOCKv<isSlu>,
		StatOrExprType::EXPRv<isSlu>
	>;
	template<AnyCfgable CfgT> using StatOrExpr = SelV<CfgT, StatOrExprV>;

	template<bool isSlu> using SoeOrBlockV = Sel<isSlu,BlockV<isSlu>,StatOrExprV<isSlu>>;
	template<AnyCfgable CfgT> using SoeOrBlock = SelV<CfgT, SoeOrBlockV>;

	template<bool isSlu> using SoeBoxOrBlockV = Sel<isSlu, BlockV<isSlu>, std::unique_ptr<StatOrExprV<isSlu>>>;
	template<AnyCfgable CfgT> using SoeBoxOrBlock = SelV<CfgT, SoeBoxOrBlockV>;

	namespace ArgsType
	{
		template<bool isSlu>
		struct EXPLISTv { ExpListV<isSlu> v; };			// "'(' [explist] ')'"
		template<AnyCfgable CfgT> using EXPLIST = SelV<CfgT, EXPLISTv>;

		template<bool isSlu>
		struct TABLEv { TableConstructorV<isSlu> v; };	// "tableconstructor"
		template<AnyCfgable CfgT> using TABLE = SelV<CfgT, TABLEv>;

		struct LITERAL { std::string v; Position end; };// "LiteralString"
	};
	template<bool isSlu>
	using ArgsV = std::variant<
		ArgsType::EXPLISTv<isSlu>,
		ArgsType::TABLEv<isSlu>,
		ArgsType::LITERAL
	>;

	template<AnyCfgable CfgT>
	using Args = SelV<CfgT, ArgsV>;

	template<bool isSlu>
	struct ArgFuncCallV
	{// funcArgs ::=  [‘:’ Name] args

		MpItmIdV<isSlu> funcName;//If empty, then no colon needed. Only used for ":xxx"
		ArgsV<isSlu> args;
	};

	template<bool isSlu>
	struct FuncCallV
	{
		std::unique_ptr<LimPrefixExprV<isSlu>> val;
		std::vector<ArgFuncCallV<isSlu>> argChain;
	};


	namespace ExprType
	{

		template<bool isSlu>
		using LIM_PREFIX_EXPv = std::unique_ptr<LimPrefixExprV<isSlu>>;	// "prefixexp"
		template<AnyCfgable CfgT> using LIM_PREFIX_EXP = SelV<CfgT, LIM_PREFIX_EXPv>;

		template<bool isSlu>
		using FUNC_CALLv = FuncCallV<isSlu>;								// "functioncall"
		template<AnyCfgable CfgT> using FUNC_CALL = SelV<CfgT, FUNC_CALLv>;

		struct OPEN_RANGE {};					// ".."

		struct LITERAL_STRING { std::string v; Position end;};	// "LiteralString"
		struct NUMERAL { double v; };							// "Numeral"

		struct NUMERAL_I64 { int64_t v; };            // "Numeral"

		//u64,i128,u128, for slu only
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

	//Slu

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
		using Struct = TableConstructorV<true>;
		struct Union {
			TableConstructorV<true> fields;
		};

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
		TypeExprDataType::Struct,
		TypeExprDataType::Union,
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
				&& std::holds_alternative<TypeExprDataType::Struct>(data);
		}
	};

	using TypePrefix = std::vector<UnOpItem>;




	//Common

	//NOTE: has overload later!!!
	template<bool isSlu>
	struct ParameterV
	{
		MpItmIdV<isSlu> name;
	};
	template<AnyCfgable CfgT>
	using Parameter = SelV<CfgT, ParameterV>;

	template<bool isSlu>
	using ParamListV = std::vector<ParameterV<isSlu>>;
	template<AnyCfgable CfgT> using ParamList = SelV<CfgT, ParamListV>;

	template<bool isSlu>
	struct BaseFunctionV
	{
		ParamListV<isSlu> params;
		BlockV<isSlu> block;
		bool hasVarArgParam = false;// do params end with '...'
	};

	template<bool isSlu>
	struct FunctionV : BaseFunctionV<isSlu>
	{};
	template<>
	struct FunctionV<true> : BaseFunctionV<true>
	{
		std::optional<TypeExpr> retType;
		OptSafety safety = OptSafety::DEFAULT;
	};

	template<AnyCfgable CfgT>
	using Function = SelV<CfgT, FunctionV>;



	template<bool isSlu,bool boxIt>
	struct BaseIfCondV
	{
		std::vector<std::pair<ExpressionV<isSlu>, SoeOrBlockV<isSlu>>> elseIfs;
		MayBox<boxIt, ExpressionV<isSlu>> cond;
		MayBox<boxIt, SoeOrBlockV<isSlu>> bl;
		std::optional<MayBox<boxIt, SoeOrBlockV<isSlu>>> elseBlock;
	};
	template<AnyCfgable CfgT, bool boxIt> 
	using BaseIfCond = Sel<CfgT::settings()&sluSyn, BaseIfCondV<false,boxIt>, BaseIfCondV<true,boxIt>>;

	namespace ExprType
	{
		using NIL = std::monostate;								// "nil"
		struct FALSE {};										// "false"
		struct TRUE {};											// "true"
		struct VARARGS {};										// "..."

		template<bool isSlu>
		struct FUNCTION_DEFv { FunctionV<isSlu> v; };				// "functiondef"
		template<AnyCfgable CfgT> using FUNCTION_DEF = SelV<CfgT, FUNCTION_DEFv>;

		template<bool isSlu>
		struct TABLE_CONSTRUCTORv { TableConstructorV<isSlu> v; };	// "tableconstructor"
		template<AnyCfgable CfgT> using TABLE_CONSTRUCTOR = SelV<CfgT, TABLE_CONSTRUCTORv>;

		//unOps is always empty for this type
		template<bool isSlu>
		struct MULTI_OPERATIONv
		{
			std::unique_ptr<ExpressionV<isSlu>> first;
			std::vector<std::pair<BinOpType, ExpressionV<isSlu>>> extra;//size>=1
		};      // "exp binop exp"
		template<AnyCfgable CfgT> using MULTI_OPERATION = SelV<CfgT, MULTI_OPERATIONv>;

		//struct UNARY_OPERATION{UnOpType,std::unique_ptr<ExpressionV<isSlu>>};     // "unop exp"	//Inlined as opt prefix

		template<bool isSlu>
		using IfCondV = BaseIfCondV<isSlu, true>;
		template<AnyCfgable CfgT> using IfCond = SelV<CfgT, IfCondV>;


		using LIFETIME = Lifetime;	// " '/' var" {'/' var"}
		using TYPE_EXPR = TypeExpr;
		using TRAIT_EXPR = TraitExpr;

		struct PAT_TYPE_PREFIX {};
	}

	template<bool isSlu>
	using ExprDataV = std::variant<
		ExprType::NIL,                  // "nil"
		ExprType::FALSE,                // "false"
		ExprType::TRUE,                 // "true"
		ExprType::NUMERAL,				// "Numeral" (e.g., a floating-point number)
		ExprType::NUMERAL_I64,			// "Numeral"

		ExprType::LITERAL_STRING,		// "LiteralString"
		ExprType::VARARGS,              // "..." (varargs)
		ExprType::FUNCTION_DEFv<isSlu>,			// "functiondef"
		ExprType::LIM_PREFIX_EXPv<isSlu>,		// "prefixexp"
		ExprType::FUNC_CALLv<isSlu>,			// "prefixexp argsThing {argsThing}"
		ExprType::TABLE_CONSTRUCTORv<isSlu>,	// "tableconstructor"

		ExprType::MULTI_OPERATIONv<isSlu>,		// "exp binop exp {binop exp}"  // added {binop exp}, cuz multi-op

		// Slu

		ExprType::IfCondV<isSlu>,

		ExprType::OPEN_RANGE,			// ".."

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


	template<bool isSlu>
	struct BaseExpressionV
	{
		ExprDataV<isSlu> data;
		Position place;
		std::vector<UnOpItem> unOps;

		BaseExpressionV() = default;
		BaseExpressionV(const BaseExpressionV&) = delete;
		BaseExpressionV(BaseExpressionV&&) = default;
		BaseExpressionV& operator=(BaseExpressionV&&) = default;
	};

	template<bool isSlu>
	struct ExpressionV : BaseExpressionV<isSlu>
	{
	};
	template<>
	struct ExpressionV<true> : BaseExpressionV<true>
	{
		SmallEnumList<PostUnOpType> postUnOps;
	};

	//Slu


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
		using DEREF = std::monostate;

		template<bool isSlu>
		struct NAMEv { MpItmIdV<isSlu> idx; };	// {funcArgs} ‘.’ Name
		template<AnyCfgable CfgT> using NAME = SelV<CfgT, NAMEv>;

		template<bool isSlu>
		struct EXPRv { ExpressionV<isSlu> idx; };	// {funcArgs} ‘[’ exp ‘]’
		template<AnyCfgable CfgT> using EXPR = SelV<CfgT, EXPRv>;
	}

	template<bool isSlu>
	struct SubVarV
	{
		std::vector<ArgFuncCallV<isSlu>> funcCalls;

		std::variant<
			SubVarType::DEREF,
			SubVarType::NAMEv<isSlu>,
			SubVarType::EXPRv<isSlu>
		> idx;
	};

	template<AnyCfgable CfgT>
	using SubVar = SelV<CfgT, SubVarV>;

	namespace BaseVarType
	{
		template<bool isSlu>
		struct NAMEv
		{
			MpItmIdV<isSlu> v;
		};
		template<AnyCfgable CfgT>
		using NAME = SelV<CfgT, NAMEv>;

		template<bool isSlu>
		struct EXPRv
		{
			ExpressionV<isSlu> start;
		};
		template<AnyCfgable CfgT> using EXPR = SelV<CfgT, EXPRv>;

	}
	template<bool isSlu>
	using BaseVarV = std::variant<
		BaseVarType::NAMEv<isSlu>,
		BaseVarType::EXPRv<isSlu>
	>;

	template<AnyCfgable CfgT>
	using BaseVar = SelV<CfgT, BaseVarV>;

	template<bool isSlu>
	struct VarV
	{
		BaseVarV<isSlu> base;
		std::vector<SubVarV<isSlu>> sub;
	};

	template<bool isSlu>
	struct AttribNameV
	{
		MpItmIdV<isSlu> name;
		std::string attrib;//empty -> no attrib
	};
	template<AnyCfgable CfgT> using AttribName = SelV<CfgT, AttribNameV>;

	namespace FieldType
	{
		template<bool isSlu>
		struct EXPR2EXPRv { ExpressionV<isSlu> idx; ExpressionV<isSlu> v; };		// "‘[’ exp ‘]’ ‘=’ exp"

		template<bool isSlu>
		struct NAME2EXPRv { MpItmIdV<isSlu> idx; ExpressionV<isSlu> v; };	// "Name ‘=’ exp"

		template<bool isSlu>
		struct EXPRv { ExpressionV<isSlu> v; };							// "exp"
	}
	namespace LimPrefixExprType
	{
		template<bool isSlu>
		struct VARv { VarV<isSlu> v; };			// "var"

		template<bool isSlu>
		struct EXPRv { ExpressionV<isSlu> v; };	// "'(' exp ')'"
	}

	template<bool isSlu>
	using AttribNameListV = std::vector<AttribNameV<isSlu>>;
	template<AnyCfgable CfgT> using AttribNameList = SelV<CfgT, AttribNameListV>;
	template<bool isSlu>
	using NameListV = std::vector<MpItmIdV<isSlu>>;
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

	template<class TyTy, bool isSlu>
	struct StructBaseV
	{
		ParamListV<isSlu> params;
		TyTy type;
		MpItmIdV<isSlu> name;
		ExportData exported = false;
	};

	namespace StatementType
	{
		using SEMICOLON = std::monostate;	// ";"

		template<bool isSlu>
		struct ASSIGNv { std::vector<VarV<isSlu>> vars; ExpListV<isSlu> exprs; };// "varlist = explist" //e.size must be > 0
		template<AnyCfgable CfgT> using ASSIGN = SelV<CfgT, ASSIGNv>;

		template<bool isSlu>
		using FUNC_CALLv = FuncCallV<isSlu>;								// "functioncall"
		template<AnyCfgable CfgT> using FUNC_CALL = SelV<CfgT, FUNC_CALLv>;

		template<bool isSlu>
		struct LABELv { MpItmIdV<isSlu> v; };		// "label"
		template<AnyCfgable CfgT> using LABEL = SelV<CfgT, LABELv>;
		struct BREAK { };
		template<bool isSlu>					// "break"
		struct GOTOv { MpItmIdV<isSlu> v; };			// "goto Name"
		template<AnyCfgable CfgT> using GOTO = SelV<CfgT, GOTOv>;

		template<bool isSlu>
		struct BLOCKv { BlockV<isSlu> bl; };							// "do block end"
		template<AnyCfgable CfgT> using BLOCK = SelV<CfgT, BLOCKv>;

		template<bool isSlu>
		struct WHILE_LOOPv { ExpressionV<isSlu> cond; BlockV<isSlu> bl; };		// "while exp do block end"
		template<AnyCfgable CfgT> using WHILE_LOOP = SelV<CfgT, WHILE_LOOPv>;

		template<bool isSlu>
		struct REPEAT_UNTILv :WHILE_LOOPv<isSlu> {};						// "repeat block until exp"
		template<AnyCfgable CfgT> using REPEAT_UNTIL = SelV<CfgT, REPEAT_UNTILv>;

		// "if exp then block {elseif exp then block} [else block] end"
		template<bool isSlu>
		using IfCondV = BaseIfCondV<isSlu, false>;
		template<AnyCfgable CfgT> using IfCond = SelV<CfgT, IfCondV>;

		// "for Name = exp , exp [, exp] do block end"
		template<bool isSlu>
		struct FOR_LOOP_NUMERICv
		{
			Sel<isSlu, MpItmIdV<isSlu>, Pat> varName;
			ExpressionV<isSlu> start;
			ExpressionV<isSlu> end;//inclusive
			std::optional<ExpressionV<isSlu>> step;
			BlockV<isSlu> bl;
		};
		template<AnyCfgable CfgT> using FOR_LOOP_NUMERIC = SelV<CfgT, FOR_LOOP_NUMERICv>;

		// "for namelist in explist do block end"
		template<bool isSlu>
		struct FOR_LOOP_GENERICv
		{
			Sel<isSlu, NameListV<isSlu>, Pat> varNames;
			Sel<isSlu, ExpListV<isSlu>, ExpressionV<isSlu>> exprs;//size must be > 0
			BlockV<isSlu> bl;
		};
		template<AnyCfgable CfgT> using FOR_LOOP_GENERIC = SelV<CfgT, FOR_LOOP_GENERICv>;

		template<bool isSlu>
		struct FuncDefBase
		{// "function funcname funcbody"    
			Position place;//Right before func-name
			MpItmIdV<isSlu> name; // name may contain dots, 1 colon if !isSlu
			FunctionV<isSlu> func;
		};
		template<bool isSlu>
		struct FUNCTION_DEFv : FuncDefBase<isSlu> {};
		template<>
		struct FUNCTION_DEFv<true> : FuncDefBase<true> 
		{
			ExportData exported = false;
		};

		template<AnyCfgable CfgT> using FUNCTION_DEF = SelV<CfgT, FUNCTION_DEFv>;

		template<bool isSlu>
		struct FNv : FUNCTION_DEFv<isSlu> {};
		template<AnyCfgable CfgT> using FN = SelV<CfgT, FNv>;

		template<bool isSlu>
		struct LOCAL_FUNCTION_DEFv :FUNCTION_DEFv<isSlu> {};
		template<AnyCfgable CfgT> using LOCAL_FUNCTION_DEF = SelV<CfgT, LOCAL_FUNCTION_DEFv>;
				// "local function Name funcbody" //n may not ^^^

		template<bool isSlu>
		struct LOCAL_ASSIGNv
		{	// "local attnamelist [= explist]" //e.size 0 means "only define, no assign"
			AttribNameListV<isSlu> names;
			ExpListV<isSlu> exprs;
		};
		template<>
		struct LOCAL_ASSIGNv<true>
		{	// "local attnamelist [= explist]" //e.size 0 means "only define, no assign"
			Pat names;
			ExpListV<true> exprs;
			ExportData exported = false;
		};
		template<AnyCfgable CfgT> using LOCAL_ASSIGN = SelV<CfgT, LOCAL_ASSIGNv>;

		// Slu

		template<bool isSlu>
		struct LETv : LOCAL_ASSIGNv<isSlu>	{};
		template<AnyCfgable CfgT> using LET = SelV<CfgT, LETv>;

		template<bool isSlu>
		struct CONSTv : LOCAL_ASSIGNv<isSlu>	{};
		template<AnyCfgable CfgT> using CONST = SelV<CfgT, CONSTv>;

		template<bool isSlu>
		struct StructV : StructBaseV<TypeExpr,isSlu> {};
		template<AnyCfgable CfgT> using Struct = SelV<CfgT, StructV>;

		template<bool isSlu>
		struct UnionV : StructBaseV<TableConstructorV<isSlu>, isSlu> {};
		template<AnyCfgable CfgT> using Union = SelV<CfgT, UnionV>;

		struct UNSAFE_LABEL {};
		struct SAFE_LABEL {};

		struct USE
		{
			MpItmIdV<true> base;//the aliased/imported thing, or modpath base
			UseVariant useVariant;
			ExportData exported=false;
		};

		template<bool isSlu>
		struct DROPv
		{
			ExpressionV<isSlu> expr;
		};
		template<AnyCfgable CfgT> using DROP = SelV<CfgT, DROPv>;

		template<bool isSlu>
		struct MOD_DEFv
		{
			MpItmIdV<isSlu> name;
			ExportData exported = false;
		};
		template<AnyCfgable CfgT> using MOD_DEF = SelV<CfgT, MOD_DEFv>;

		template<bool isSlu>
		struct MOD_DEF_INLINEv
		{ 
			MpItmIdV<isSlu> name;
			BlockV<isSlu> bl;
			ExportData exported = false;
		};
		template<AnyCfgable CfgT> using MOD_DEF_INLINE = SelV<CfgT, MOD_DEF_INLINEv>;

	};

	template<bool isSlu>
	using StatementDataV = std::variant<
		StatementType::SEMICOLON,				// ";"

		StatementType::ASSIGNv<isSlu>,			// "varlist = explist"
		StatementType::LOCAL_ASSIGNv<isSlu>,	// "local attnamelist [= explist]"
		StatementType::LETv<isSlu>,	// "let pat [= explist]"
		StatementType::CONSTv<isSlu>,	// "const pat [= explist]"

		StatementType::FUNC_CALLv<isSlu>,		// "functioncall"
		StatementType::LABELv<isSlu>,			// "label"
		StatementType::BREAK,					// "break"
		StatementType::GOTOv<isSlu>,			// "goto Name"
		StatementType::BLOCKv<isSlu>,			// "do block end"
		StatementType::WHILE_LOOPv<isSlu>,		// "while exp do block end"
		StatementType::REPEAT_UNTILv<isSlu>,	// "repeat block until exp"

		StatementType::IfCondV<isSlu>,	// "if exp then block {elseif exp then block} [else block] end"

		StatementType::FOR_LOOP_NUMERICv<isSlu>,	// "for Name = exp , exp [, exp] do block end"
		StatementType::FOR_LOOP_GENERICv<isSlu>,	// "for namelist in explist do block end"

		StatementType::FUNCTION_DEFv<isSlu>,		// "function funcname funcbody"
		StatementType::FNv<isSlu>,					// "fn funcname funcbody"
		StatementType::LOCAL_FUNCTION_DEFv<isSlu>,	// "local function Name funcbody"

		StatementType::StructV<isSlu>,
		StatementType::UnionV<isSlu>,

		StatementType::UNSAFE_LABEL,	// ::: unsafe :
		StatementType::SAFE_LABEL,		// ::: safe :

		StatementType::DROPv<isSlu>,	// "drop" Name
		StatementType::USE,				// "use" ...
		StatementType::MOD_DEFv<isSlu>,		// "mod" Name
		StatementType::MOD_DEF_INLINEv<isSlu>	// "mod" Name "as" "{" block "}"
	> ;

	template<AnyCfgable CfgT>
	using StatementData = SelV<CfgT, StatementDataV>;

	template<bool isSlu>
	struct StatementV
	{
		StatementDataV<isSlu> data;
		Position place;

		StatementV() = default;
		StatementV(StatementDataV<isSlu>&& data) :data(std::move(data)) {}
		StatementV(const StatementV&) = delete;
		StatementV(StatementV&&) = default;
		StatementV& operator=(StatementV&&) = default;
	};


	template<bool isSlu>
	struct ParsedFileV
	{
		//TypeList types
		BlockV<isSlu> code;
	};
	template<AnyCfgable CfgT>
	using ParsedFile = SelV<CfgT, ParsedFileV>;
}