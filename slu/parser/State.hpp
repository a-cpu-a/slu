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

#define Slu_DEF_CFG(_Name) template<AnyCfgable CfgT> using _Name = SelV<CfgT, _Name ## V>
#define Slu_DEF_CFG_CAPS(_NAME) template<AnyCfgable CfgT> using _NAME = SelV<CfgT, _NAME ## v>

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
	Slu_DEF_CFG(Statement);

	template<bool isSlu> struct ExpressionV;
	Slu_DEF_CFG(Expression);

	template<bool isSlu> struct VarV;
	Slu_DEF_CFG(Var);

	namespace FieldType {
		template<bool isSlu> struct EXPR2EXPRv;
		Slu_DEF_CFG_CAPS(EXPR2EXPR);

		template<bool isSlu> struct NAME2EXPRv;
		Slu_DEF_CFG_CAPS(NAME2EXPR);

		template<bool isSlu> struct EXPRv;
		Slu_DEF_CFG_CAPS(EXPR);
	}
	namespace LimPrefixExprType
	{
		template<bool isSlu> struct VARv;			// "var"
		Slu_DEF_CFG_CAPS(VAR);

		template<bool isSlu> struct EXPRv;	// "'(' exp ')'"
		Slu_DEF_CFG_CAPS(EXPR);
	}
	template<bool isSlu>
	using LimPrefixExprV = std::variant<
		LimPrefixExprType::VARv<isSlu>,
		LimPrefixExprType::EXPRv<isSlu>
	>;
	Slu_DEF_CFG(LimPrefixExpr);

	template<bool isSlu> struct ArgFuncCallV;
	Slu_DEF_CFG(ArgFuncCall);

	template<bool isSlu> struct FuncCallV;
	Slu_DEF_CFG(FuncCall);

	template<bool isSlu>
	using ExpListV = std::vector<ExpressionV<isSlu>>;
	Slu_DEF_CFG(ExpList);

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
	Slu_DEF_CFG(Field);

	// ‘{’ [fieldlist] ‘}’
	template<bool isSlu>
	using TableConstructorV = std::vector<FieldV<isSlu>>;
	Slu_DEF_CFG(TableConstructor);




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
	Slu_DEF_CFG(Block);

	namespace SoeType
	{
		template<bool isSlu>
		using BLOCKv = BlockV<isSlu>;
		Slu_DEF_CFG_CAPS(BLOCK);

		template<bool isSlu>
		using EXPRv = std::unique_ptr<ExpressionV<isSlu>>;
		Slu_DEF_CFG_CAPS(EXPR);
	}
	template<bool isSlu>
	using SoeV = std::variant<
		SoeType::BLOCKv<isSlu>,
		SoeType::EXPRv<isSlu>
	>;
	Slu_DEF_CFG(Soe);

	template<bool isSlu> using SoeOrBlockV = Sel<isSlu,BlockV<isSlu>,SoeV<isSlu>>;
	Slu_DEF_CFG(SoeOrBlock);

	template<bool isSlu> using SoeBoxOrBlockV = Sel<isSlu, BlockV<isSlu>, std::unique_ptr<SoeV<isSlu>>>;
	Slu_DEF_CFG(SoeBoxOrBlock);

	namespace ArgsType
	{
		template<bool isSlu>
		struct EXPLISTv { ExpListV<isSlu> v; };			// "'(' [explist] ')'"
		Slu_DEF_CFG_CAPS(EXPLIST);

		template<bool isSlu>
		struct TABLEv { TableConstructorV<isSlu> v; };	// "tableconstructor"
		Slu_DEF_CFG_CAPS(TABLE);

		struct LITERAL { std::string v; Position end; };// "LiteralString"
	};
	template<bool isSlu>
	using ArgsV = std::variant<
		ArgsType::EXPLISTv<isSlu>,
		ArgsType::TABLEv<isSlu>,
		ArgsType::LITERAL
	>;
	Slu_DEF_CFG(Args);

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
		Slu_DEF_CFG_CAPS(LIM_PREFIX_EXP);

		template<bool isSlu>
		using FUNC_CALLv = FuncCallV<isSlu>;								// "functioncall"
		Slu_DEF_CFG_CAPS(FUNC_CALL);

		struct OPEN_RANGE {};					// ".."

		struct LITERAL_STRING { std::string v; Position end;};	// "LiteralString"
		struct NUMERAL { double v; };							// "Numeral"

		struct NUMERAL_I64 { int64_t v; };            // "Numeral"

		//u64,i128,u128, for slu only
		struct NUMERAL_U64 { uint64_t v; };						// "Numeral"
		struct NUMERAL_U128 { // "Numeral"
			uint64_t lo = 0; 
			uint64_t hi = 0; 

			NUMERAL_U128 operator+(NUMERAL_U128 o) const
			{
				NUMERAL_U128 res = *this;
				res.lo += o.lo;
				res.hi += o.hi;
				if (res.lo < lo)//overflow check
					res.hi++;
				return res;
			}
			NUMERAL_U128 shift(uint8_t count) const
			{
				NUMERAL_U128 res = *this;
				res.hi <<= count;
				res.hi |= lo >> (64 - count);
				res.lo <<= count;
				return res;
			}

		};
		struct NUMERAL_I128 :NUMERAL_U128 {};					// "Numeral"
	}

	struct TupleName
	{
		uint64_t lo = 0; uint64_t hi = 0;
		constexpr TupleName() = default;
		constexpr TupleName(ExprType::NUMERAL_I64 v)
			:lo(v.v) {}
		constexpr TupleName(ExprType::NUMERAL_U64 v) 
			:lo(v.v) {}
		constexpr TupleName(ExprType::NUMERAL_I128 v)
			:lo(v.lo),hi(v.hi) {}						   
		constexpr TupleName(ExprType::NUMERAL_U128 v) 
			:lo(v.lo),hi(v.hi) {}
	};

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
		std::vector<UnOpItem> unOps;
		SmallEnumList<PostUnOpType> postUnOps;//TODO: parse this!
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
	Slu_DEF_CFG(Parameter);

	template<bool isSlu>
	using ParamListV = std::vector<ParameterV<isSlu>>;
	Slu_DEF_CFG(ParamList);

	template<bool isSlu>
	struct FunctionInfoV
	{
		ParamListV<isSlu> params;
		bool hasVarArgParam = false;// do params end with '...'
	};
	template<>
	struct FunctionInfoV<true>
	{
		std::string abi;
		ParamListV<true> params;
		std::optional<TypeExpr> retType;
		bool hasVarArgParam = false;// do params end with '...'
		OptSafety safety = OptSafety::DEFAULT;
	};
	Slu_DEF_CFG(FunctionInfo);

	template<bool isSlu>
	struct FunctionV : FunctionInfoV<isSlu>
	{
		BlockV<isSlu> block;
	};
	Slu_DEF_CFG(Function);



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
		Slu_DEF_CFG_CAPS(FUNCTION_DEF);

		template<bool isSlu>
		struct TABLE_CONSTRUCTORv { TableConstructorV<isSlu> v; };	// "tableconstructor"
		Slu_DEF_CFG_CAPS(TABLE_CONSTRUCTOR);

		//unOps is always empty for this type
		template<bool isSlu>
		struct MULTI_OPERATIONv
		{
			std::unique_ptr<ExpressionV<isSlu>> first;
			std::vector<std::pair<BinOpType, ExpressionV<isSlu>>> extra;//size>=1
		};      // "exp binop exp"
		Slu_DEF_CFG_CAPS(MULTI_OPERATION);

		//struct UNARY_OPERATION{UnOpType,std::unique_ptr<ExpressionV<isSlu>>};     // "unop exp"	//Inlined as opt prefix

		template<bool isSlu>
		using IfCondV = BaseIfCondV<isSlu, true>;
		Slu_DEF_CFG(IfCond);


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
	Slu_DEF_CFG(ExprData);


	template<bool isSlu>
	struct BaseExpressionV
	{
		ExprDataV<isSlu> data;
		Position place;
		std::vector<UnOpItem> unOps;//TODO: for lua, use small op list

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

	template<bool isSlu = true>
	using NdPatV = ExpressionV<isSlu>;
	Slu_DEF_CFG(NdPat);

	namespace DestrSpecType
	{
		template<bool isSlu>
		using SpatV = parse::NdPatV<isSlu>;
		Slu_DEF_CFG(Spat);
		using Type = TypeExpr;
		using Prefix = TypePrefix;
	}
	template<bool isSlu=true>
	using DestrSpecV = std::variant<
		DestrSpecType::SpatV<isSlu>,
		DestrSpecType::Type,
		DestrSpecType::Prefix
	>;
	Slu_DEF_CFG(DestrSpec);
	namespace DestrPatType
	{
		using Any = std::monostate;
		template<bool isSlu> struct FieldsV;
		Slu_DEF_CFG(Fields);
		template<bool isSlu> struct ListV;
		Slu_DEF_CFG(List);

		template<bool isSlu>struct NameV;
		Slu_DEF_CFG(Name);
		template<bool isSlu>struct NameRestrictV;
		Slu_DEF_CFG(NameRestrict);
	}

	namespace PatType
	{
		//x or y or z
		template<bool isSlu>
		using SimpleV = NdPatV<isSlu>;
		Slu_DEF_CFG(Simple);

		using DestrAny = DestrPatType::Any;

		template<bool isSlu>
		using DestrFieldsV = DestrPatType::FieldsV<isSlu>;
		Slu_DEF_CFG(DestrFields);
		template<bool isSlu>
		using DestrListV = DestrPatType::ListV<isSlu>;
		Slu_DEF_CFG(DestrList);

		template<bool isSlu>
		using DestrNameV = DestrPatType::NameV<isSlu>;
		Slu_DEF_CFG(DestrName);

		template<bool isSlu>
		using DestrNameRestrictV = DestrPatType::NameRestrictV<isSlu>;
		Slu_DEF_CFG(DestrNameRestrict);
	}
	template<typename T>
	concept AnyCompoundDestr =
		std::same_as<std::remove_cv_t<T>, DestrPatType::FieldsV<true>>
		|| std::same_as<std::remove_cv_t<T>, DestrPatType::ListV<true>>;


	template<bool isSlu>
	using PatV = std::variant<
		PatType::DestrAny,

		PatType::SimpleV<isSlu>,

		PatType::DestrFieldsV<isSlu>,
		PatType::DestrListV<isSlu>,

		PatType::DestrNameV<isSlu>,
		PatType::DestrNameRestrictV<isSlu>
	>;
	Slu_DEF_CFG(Pat);

	template<bool isSlu>
	struct DestrFieldV
	{
		MpItmIdV<isSlu> name;
		PatV<isSlu> pat;
	};
	Slu_DEF_CFG(DestrField);
	namespace DestrPatType
	{
		template<bool isSlu>
		struct FieldsV
		{
			DestrSpecV<isSlu> spec;
			bool extraFields : 1 = false;
			std::vector<DestrFieldV<isSlu>> items;
			MpItmIdV<isSlu> name;//May be empty
		};
		template<bool isSlu>
		struct ListV
		{
			DestrSpecV<isSlu> spec;
			bool extraFields : 1 = false;
			std::vector<PatV<isSlu>> items;
			MpItmIdV<isSlu> name;//May be empty
		};

		template<bool isSlu>
		struct NameV
		{
			MpItmIdV<isSlu> name;
			DestrSpecV<isSlu> spec;
		};
		template<bool isSlu>
		struct NameRestrictV : NameV<isSlu>
		{
			NdPatV<isSlu> restriction;
		};
	}

	template<>
	struct ParameterV<true>
	{
		PatV<true> name;
	};
	struct ___PatHack : PatV<true> {};

	//Common

	namespace SubVarType
	{
		using DEREF = std::monostate;

		template<bool isSlu>
		struct NAMEv { MpItmIdV<isSlu> idx; };	// {funcArgs} ‘.’ Name
		Slu_DEF_CFG_CAPS(NAME);

		template<bool isSlu>
		struct EXPRv { ExpressionV<isSlu> idx; };	// {funcArgs} ‘[’ exp ‘]’
		Slu_DEF_CFG_CAPS(EXPR);
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

	Slu_DEF_CFG(SubVar);

	namespace BaseVarType
	{
		using Root = std::monostate;// ":>" // modpath root

		template<bool isSlu>
		struct NAMEv
		{
			MpItmIdV<isSlu> v;
		};
		Slu_DEF_CFG_CAPS(NAME);

		template<bool isSlu>
		struct EXPRv
		{
			ExpressionV<isSlu> start;
		};
		Slu_DEF_CFG_CAPS(EXPR);

	}
	template<bool isSlu>
	using BaseVarV = std::variant<
		BaseVarType::Root,
		BaseVarType::NAMEv<isSlu>,
		BaseVarType::EXPRv<isSlu>
	>;
	Slu_DEF_CFG(BaseVar);

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
	Slu_DEF_CFG(AttribName);

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
	Slu_DEF_CFG(AttribNameList);
	template<bool isSlu>
	using NameListV = std::vector<MpItmIdV<isSlu>>;
	Slu_DEF_CFG(NameList);

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
		Slu_DEF_CFG_CAPS(ASSIGN);

		template<bool isSlu>
		using FUNC_CALLv = FuncCallV<isSlu>;								// "functioncall"
		Slu_DEF_CFG_CAPS(FUNC_CALL);

		template<bool isSlu>
		struct LABELv { MpItmIdV<isSlu> v; };		// "label"
		Slu_DEF_CFG_CAPS(LABEL);
		struct BREAK { };
		template<bool isSlu>					// "break"
		struct GOTOv { MpItmIdV<isSlu> v; };			// "goto Name"
		Slu_DEF_CFG_CAPS(GOTO);

		template<bool isSlu>
		struct BLOCKv { BlockV<isSlu> bl; };							// "do block end"
		Slu_DEF_CFG_CAPS(BLOCK);

		template<bool isSlu>
		struct WHILE_LOOPv { ExpressionV<isSlu> cond; BlockV<isSlu> bl; };		// "while exp do block end"
		Slu_DEF_CFG_CAPS(WHILE_LOOP);

		template<bool isSlu>
		struct REPEAT_UNTILv :WHILE_LOOPv<isSlu> {};						// "repeat block until exp"
		Slu_DEF_CFG_CAPS(REPEAT_UNTIL);

		// "if exp then block {elseif exp then block} [else block] end"
		template<bool isSlu>
		using IfCondV = BaseIfCondV<isSlu, false>;
		Slu_DEF_CFG(IfCond);

		// "for Name = exp , exp [, exp] do block end"
		template<bool isSlu>
		struct FOR_LOOP_NUMERICv
		{
			Sel<isSlu, MpItmIdV<isSlu>, PatV<true>> varName;
			ExpressionV<isSlu> start;
			ExpressionV<isSlu> end;//inclusive
			std::optional<ExpressionV<isSlu>> step;
			BlockV<isSlu> bl;
		};
		Slu_DEF_CFG_CAPS(FOR_LOOP_NUMERIC);

		// "for namelist in explist do block end"
		template<bool isSlu>
		struct FOR_LOOP_GENERICv
		{
			Sel<isSlu, NameListV<isSlu>, PatV<true>> varNames;
			Sel<isSlu, ExpListV<isSlu>, ExpressionV<isSlu>> exprs;//size must be > 0
			BlockV<isSlu> bl;
		};
		Slu_DEF_CFG_CAPS(FOR_LOOP_GENERIC);

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
		Slu_DEF_CFG_CAPS(FUNCTION_DEF);

		template<bool isSlu>
		struct FNv : FUNCTION_DEFv<isSlu> {};
		Slu_DEF_CFG_CAPS(FN);

		template<bool isSlu>
		struct LOCAL_FUNCTION_DEFv :FUNCTION_DEFv<isSlu> {};
		Slu_DEF_CFG_CAPS(LOCAL_FUNCTION_DEF);
				// "local function Name funcbody" //n may not ^^^


		template<bool isSlu>
		struct FunctionDeclV : FunctionInfoV<isSlu>
		{
			Position place;//Right before func-name
			MpItmIdV<isSlu> name;
			ExportData exported = false;
		};
		Slu_DEF_CFG(FunctionDecl);

		template<bool isSlu>
		struct FnDeclV : FunctionDeclV<isSlu> {};
		Slu_DEF_CFG(FnDecl);

		template<bool isSlu>
		struct LOCAL_ASSIGNv
		{	// "local attnamelist [= explist]" //e.size 0 means "only define, no assign"
			AttribNameListV<isSlu> names;
			ExpListV<isSlu> exprs;
		};
		template<>
		struct LOCAL_ASSIGNv<true>
		{	// "local attnamelist [= explist]" //e.size 0 means "only define, no assign"
			PatV<true> names;
			ExpListV<true> exprs;
			ExportData exported = false;
		};
		Slu_DEF_CFG_CAPS(LOCAL_ASSIGN);

		// Slu

		template<bool isSlu>
		struct LETv : LOCAL_ASSIGNv<isSlu>	{};
		Slu_DEF_CFG_CAPS(LET);

		template<bool isSlu>
		struct CONSTv : LOCAL_ASSIGNv<isSlu>	{};
		Slu_DEF_CFG_CAPS(CONST);

		template<bool isSlu>
		struct StructV : StructBaseV<TypeExpr,isSlu> {};
		Slu_DEF_CFG(Struct);

		template<bool isSlu>
		struct UnionV : StructBaseV<TableConstructorV<isSlu>, isSlu> {};
		Slu_DEF_CFG(Union);

		template<bool isSlu>
		struct ExternBlockV {
			BlockV<isSlu> bl;
			std::string abi;
			Position abiEnd;
			OptSafety safety = OptSafety::DEFAULT;
		};
		Slu_DEF_CFG(ExternBlock);

		template<bool isSlu>
		struct UnsafeBlockV { BlockV<isSlu> bl; };	// "unsafe {...}"
		Slu_DEF_CFG(UnsafeBlock);

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
		Slu_DEF_CFG_CAPS(DROP);

		template<bool isSlu>
		struct MOD_DEFv
		{
			MpItmIdV<isSlu> name;
			ExportData exported = false;
		};
		Slu_DEF_CFG_CAPS(MOD_DEF);

		template<bool isSlu>
		struct MOD_DEF_INLINEv
		{ 
			MpItmIdV<isSlu> name;
			BlockV<isSlu> bl;
			ExportData exported = false;
		};
		Slu_DEF_CFG_CAPS(MOD_DEF_INLINE);

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

		StatementType::LOCAL_FUNCTION_DEFv<isSlu>,	// "local function Name funcbody"
		StatementType::FUNCTION_DEFv<isSlu>,		// "function funcname funcbody"
		StatementType::FNv<isSlu>,					// "fn funcname funcbody"

		StatementType::FunctionDeclV<isSlu>,
		StatementType::FnDeclV<isSlu>,

		StatementType::StructV<isSlu>,
		StatementType::UnionV<isSlu>,

		StatementType::ExternBlockV<isSlu>,

		StatementType::UnsafeBlockV<isSlu>,
		StatementType::UNSAFE_LABEL,	// ::: unsafe :
		StatementType::SAFE_LABEL,		// ::: safe :

		StatementType::DROPv<isSlu>,	// "drop" Name
		StatementType::USE,				// "use" ...
		StatementType::MOD_DEFv<isSlu>,		// "mod" Name
		StatementType::MOD_DEF_INLINEv<isSlu>	// "mod" Name "as" "{" block "}"
	>;
	Slu_DEF_CFG(StatementData);

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
	Slu_DEF_CFG(ParsedFile);
}