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

namespace slu::parse
{

	template<AnyCfgable CfgT, template<bool> class T>
	using SelV = T<CfgT::settings()& sluSyn>;

	template<bool isSlu, class T, class SlT>
	using Sel = std::conditional_t<isSlu, SlT, T>;

#define Slu_DEF_CFG(_Name) template<AnyCfgable CfgT> using _Name = SelV<CfgT, _Name ## V>
#define Slu_DEF_CFG2(_Name,_ArgName) template<AnyCfgable CfgT,bool _ArgName> using _Name =Sel<CfgT::settings()& sluSyn, _Name ## V<false, _ArgName>, _Name ## V<true, _ArgName>>
#define Slu_DEF_CFG_CAPS(_NAME) template<AnyCfgable CfgT> using _NAME = SelV<CfgT, _NAME ## v>

	template<AnyCfgable Cfg, size_t TOK_SIZE, size_t TOK_SIZE2>
	consteval const auto& sel(const char(&tok)[TOK_SIZE], const char(&sluTok)[TOK_SIZE2])
	{
		if constexpr (Cfg::settings() & sluSyn)
			return sluTok;
		else
			return tok;
	}
	template<bool boxed, class T>
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
	template<bool boxed, class T>
	constexpr auto mayBoxFrom(T&& v)
	{
		if constexpr (boxed)
			return MayBox<true, T>(std::make_unique<T>(std::move(v)));
		else
			return MayBox<false, T>(std::move(v));
	}
	template<class T>
	constexpr MayBox<false, T> wontBox(T&& v) {
		return MayBox<false, T>(std::move(v));
	}

	//Mp ref
	template<AnyCfgable CfgT> using MpItmId = SelV<CfgT, lang::MpItmIdV>;



	//Forward declare

	template<bool isSlu> struct StatementV;
	Slu_DEF_CFG(Statement);

	template<bool isSlu> struct ExprV;
	Slu_DEF_CFG(Expr);
	template<bool isSlu>
	using BoxExprV = std::unique_ptr<ExprV<isSlu>>;

	template<bool isSlu> struct VarV;
	Slu_DEF_CFG(Var);

	namespace FieldType
	{
		//For lua only!
		template<bool isSlu> struct EXPR2EXPRv;
		Slu_DEF_CFG_CAPS(EXPR2EXPR);

		template<bool isSlu> struct NAME2EXPRv;
		Slu_DEF_CFG_CAPS(NAME2EXPR);

		using parse::ExprV;
		using parse::Expr;
	}
	namespace LimPrefixExprType
	{
		template<bool isSlu> struct VARv;
		Slu_DEF_CFG_CAPS(VAR);

		using parse::ExprV;
		using parse::Expr;
	}
	template<bool isSlu>
	using LimPrefixExprV = std::variant<
		LimPrefixExprType::VARv<isSlu>,
		LimPrefixExprType::ExprV<isSlu>
	>;
	Slu_DEF_CFG(LimPrefixExpr);

	template<bool isSlu> struct ArgFuncCallV;
	Slu_DEF_CFG(ArgFuncCall);

	template<bool isSlu> struct FuncCallV;
	Slu_DEF_CFG(FuncCall);

	template<bool isSlu>
	using ExprListV = std::vector<ExprV<isSlu>>;
	Slu_DEF_CFG(ExprList);

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


	namespace FieldType { using NONE = std::monostate; }

	template<bool isSlu>
	using FieldV = std::variant<
		FieldType::NONE,// Here, so variant has a default value (DO NOT USE)

		FieldType::EXPR2EXPRv<isSlu>, // "'[' exp ']' = exp"
		FieldType::NAME2EXPRv<isSlu>, // "Name = exp"
		FieldType::ExprV<isSlu>       // "exp"
	>;
	Slu_DEF_CFG(Field);

	// ‘{’ [fieldlist] ‘}’
	template<bool isSlu>
	using TableV = std::vector<FieldV<isSlu>>;
	Slu_DEF_CFG(Table);

	template<bool isSlu>
	using StatListV = std::vector<StatementV<isSlu>>;
	Slu_DEF_CFG(StatList);


	struct LocalId
	{
		size_t v = SIZE_MAX;
		constexpr bool empty() const { return v == SIZE_MAX; }
		constexpr static LocalId newEmpty() { return LocalId(); }
	};
	template<bool isSlu, bool isLocal>
	using LocalOrNameV = Sel<isLocal, MpItmIdV<isSlu>, LocalId>;
	Slu_DEF_CFG2(LocalOrName, isLocal);
	template<bool isSlu>
	using LocalsV = std::vector<MpItmIdV<isSlu>>;
	Slu_DEF_CFG(Locals);

	template<bool isSlu>
	using DynLocalOrNameV = Sel<isSlu, MpItmIdV<false>, std::variant<MpItmIdV<true>, LocalId>>;

	template<bool isSlu>
	struct BlockV
	{
		StatListV<isSlu> statList;
		ExprListV<isSlu> retExprs;//Special, may contain 0 elements (even with hadReturn)

		lang::ModPathId mp;

		Position start;
		Position end;

		bool hadReturn = false;

		bool empty() const {
			return !hadReturn && statList.empty();
		}

		BlockV() = default;
		BlockV(const BlockV&) = delete;
		BlockV(BlockV&&) = default;
		BlockV& operator=(BlockV&&) = default;
	};
	Slu_DEF_CFG(Block);

	namespace SoeType
	{
		template<bool isSlu>
		using BlockV = BlockV<isSlu>;
		Slu_DEF_CFG(Block);

		template<bool isSlu>
		using ExprV = BoxExprV<isSlu>;
		Slu_DEF_CFG(Expr);
	}
	template<bool isSlu>
	using SoeV = std::variant<
		SoeType::BlockV<isSlu>,
		SoeType::ExprV<isSlu>
	>;
	Slu_DEF_CFG(Soe);

	template<bool isSlu> using SoeOrBlockV = Sel<isSlu, BlockV<isSlu>, SoeV<isSlu>>;
	Slu_DEF_CFG(SoeOrBlock);

	template<bool isSlu> using SoeBoxOrBlockV = Sel<isSlu, BlockV<isSlu>, std::unique_ptr<SoeV<isSlu>>>;
	Slu_DEF_CFG(SoeBoxOrBlock);

	namespace ArgsType
	{
		template<bool isSlu>
		struct EXPLISTv { ExprListV<isSlu> v; };			// "'(' [explist] ')'"
		Slu_DEF_CFG_CAPS(EXPLIST);

		template<bool isSlu>
		struct TABLEv { TableV<isSlu> v; };	// "tableconstructor"
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
		using LimPrefixExprV = std::unique_ptr<parse::LimPrefixExprV<isSlu>>;	// "prefixexp"
		Slu_DEF_CFG(LimPrefixExpr);

		using parse::FuncCallV;
		using parse::FuncCall;

		struct OpenRange {};					// ".."

		struct String { std::string v; Position end; };	// "LiteralString"	

		// "Numeral"
		using F64 = double;
		using I64 = int64_t;

		//u64,i128,u128, for slu only
		using U64 = uint64_t;
		struct U128
		{
			uint64_t lo = 0;
			uint64_t hi = 0;

			U128 operator+(U128 o) const
			{
				U128 res = *this;
				res.lo += o.lo;
				res.hi += o.hi;
				if (res.lo < lo)//overflow check
					res.hi++;
				return res;
			}
			U128 shift(uint8_t count) const
			{
				U128 res = *this;
				res.hi <<= count;
				res.hi |= lo >> (64 - count);
				res.lo <<= count;
				return res;
			}

		};
		struct I128 :U128 {};					// "Numeral"
	}

	struct TupleName
	{
		uint64_t lo = 0; uint64_t hi = 0;
		constexpr TupleName() = default;
		constexpr TupleName(ExprType::I64 v)
			:lo(v) {}
		constexpr TupleName(ExprType::U64 v)
			: lo(v) {}
		constexpr TupleName(ExprType::I128 v)
			: lo(v.lo), hi(v.hi) {}
		constexpr TupleName(ExprType::U128 v)
			: lo(v.lo), hi(v.hi) {}
	};

	template<class T>
	concept Any128BitInt =
		std::same_as<T, ExprType::U128>
		|| std::same_as<T, ExprType::I128>;
	template<class T>
	concept Any64BitInt =
		std::same_as<T, ExprType::U64>
		|| std::same_as<T, ExprType::I64>;

	using Lifetime = std::vector<MpItmIdV<true>>;
	struct UnOpItem
	{
		Lifetime life;
		UnOpType type;
	};

	//Slu

	namespace TraitExprItemType
	{
		using LimPrefixExpr = std::unique_ptr<LimPrefixExprV<true>>;
		using FuncCall = FuncCallV<true>;
	}
	using TraitExprItem = std::variant<
		TraitExprItemType::LimPrefixExpr,
		TraitExprItemType::FuncCall
	>;
	struct TraitExpr
	{
		std::vector<TraitExprItem> traitCombo;
		Position place;
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
		LocalsV<true> local2Mp;
		ParamListV<true> params;
		std::optional<BoxExprV<true>> retType;
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



	template<bool isSlu, bool boxIt>
	struct BaseIfCondV
	{
		std::vector<std::pair<ExprV<isSlu>, SoeOrBlockV<isSlu>>> elseIfs;
		MayBox<boxIt, ExprV<isSlu>> cond;
		MayBox<boxIt, SoeOrBlockV<isSlu>> bl;
		std::optional<MayBox<boxIt, SoeOrBlockV<isSlu>>> elseBlock;
	};
	Slu_DEF_CFG2(BaseIfCond, boxIt);

	namespace ExprType
	{
		using Nil = std::monostate;								// "nil"
		struct False {};										// "false"
		struct True {};											// "true"
		struct VarArgs {};										// "..."

		using parse::FunctionV;
		using parse::Function;
		
		using parse::TableV;
		using parse::Table;

		//unOps is always empty for this type
		template<bool isSlu>
		struct MultiOpV
		{
			BoxExprV<isSlu> first;
			std::vector<std::pair<BinOpType, ExprV<isSlu>>> extra;//size>=1
		};      // "exp binop exp"
		Slu_DEF_CFG(MultiOp);

		template<bool isSlu>
		using IfCondV = BaseIfCondV<isSlu, true>;
		Slu_DEF_CFG(IfCond);


		using parse::Lifetime;	// " '/' var" {'/' var"}
		using parse::TraitExpr;

		struct PatTypePrefix {};


		struct Inferr {};
		struct Union
		{
			TableV<true> fields;
		};
		struct FnType
		{
			BoxExprV<true> argType;
			BoxExprV<true> retType;
			OptSafety safety = OptSafety::DEFAULT;
		};
		struct Dyn
		{
			TraitExpr expr;
		};
		struct Impl
		{
			TraitExpr expr;
		};
		using Slice = BoxExprV<true>;
		struct Err
		{
			BoxExprV<true> err;
		};
	}

	template<bool isSlu>
	using ExprDataV = std::variant <
		ExprType::Nil,                  // "nil"
		ExprType::False,                // "false"
		ExprType::True,                 // "true"
		ExprType::F64,				// "Numeral" (e.g., a floating-point number)
		ExprType::I64,			// "Numeral"

		ExprType::String,		// "LiteralString"
		ExprType::VarArgs,              // "..." (varargs)
		ExprType::FunctionV<isSlu>,			// "functiondef"
		ExprType::LimPrefixExprV<isSlu>,		// "prefixexp"
		ExprType::FuncCallV<isSlu>,			// "prefixexp argsThing {argsThing}"
		ExprType::TableV<isSlu>,	// "tableconstructor"

		ExprType::MultiOpV<isSlu>,		// "exp binop exp {binop exp}"  // added {binop exp}, cuz multi-op

		// Slu

		ExprType::IfCondV<isSlu>,

		ExprType::OpenRange,			// ".."

		ExprType::U64,			// "Numeral"
		ExprType::I128,			// "Numeral"
		ExprType::U128,			// "Numeral"

		ExprType::Lifetime,
		ExprType::TraitExpr,

		ExprType::PatTypePrefix,

		// types

		ExprType::Inferr,
		ExprType::Union,
		ExprType::Dyn,
		ExprType::Impl,
		ExprType::Slice,
		ExprType::Err,
		ExprType::FnType
	> ;
	Slu_DEF_CFG(ExprData);


	template<bool isSlu>
	struct BaseExprV
	{
		ExprDataV<isSlu> data;
		Position place;
		std::vector<UnOpItem> unOps;//TODO: for lua, use small op list

		BaseExprV() = default;
		BaseExprV(ExprDataV<isSlu>&& data):data(std::move(data)) {}
		BaseExprV(ExprDataV<isSlu>&& data,Position place):data(std::move(data)),place(place) {}

		BaseExprV(const BaseExprV&) = delete;
		BaseExprV(BaseExprV&&) = default;
		BaseExprV& operator=(BaseExprV&&) = default;
	};

	template<bool isSlu>
	struct ExprV : BaseExprV<isSlu>
	{};
	template<>
	struct ExprV<true> : BaseExprV<true>
	{
		SmallEnumList<PostUnOpType> postUnOps;

		bool isBasicStruct() const {
			if(!this->unOps.empty() || !this->postUnOps.empty())
				return false;
			return std::holds_alternative<ExprType::TableV<true>>(this->data);
		}
	};

	//Slu


	// match patterns

	template<bool isSlu>
	using NdPatV = ExprV<isSlu>;
	Slu_DEF_CFG(NdPat);

	namespace DestrSpecType
	{
		template<bool isSlu>
		using SpatV = parse::NdPatV<isSlu>;
		Slu_DEF_CFG(Spat);
		using Prefix = TypePrefix;
	}
	template<bool isSlu>
	using DestrSpecV = std::variant<
		DestrSpecType::SpatV<isSlu>,
		DestrSpecType::Prefix
	>;
	Slu_DEF_CFG(DestrSpec);
	namespace DestrPatType
	{
		using Any = std::monostate;
		template<bool isSlu, bool isLocal> struct FieldsV;
		Slu_DEF_CFG2(Fields, isLocal);
		template<bool isSlu, bool isLocal> struct ListV;
		Slu_DEF_CFG2(List, isLocal);

		template<bool isSlu, bool isLocal>struct NameV;
		Slu_DEF_CFG2(Name, isLocal);
		template<bool isSlu, bool isLocal>struct NameRestrictV;
		Slu_DEF_CFG2(NameRestrict, isLocal);
	}

	namespace PatType
	{
		//x or y or z
		template<bool isSlu>
		using SimpleV = NdPatV<isSlu>;
		Slu_DEF_CFG(Simple);

		using DestrAny = DestrPatType::Any;

		template<bool isSlu,bool isLocal>
		using DestrFieldsV = DestrPatType::FieldsV<isSlu, isLocal>;
		Slu_DEF_CFG2(DestrFields, isLocal);
		template<bool isSlu, bool isLocal>
		using DestrListV = DestrPatType::ListV<isSlu, isLocal>;
		Slu_DEF_CFG2(DestrList, isLocal);

		template<bool isSlu, bool isLocal>
		using DestrNameV = DestrPatType::NameV<isSlu, isLocal>;
		Slu_DEF_CFG2(DestrName, isLocal);

		template<bool isSlu, bool isLocal>
		using DestrNameRestrictV = DestrPatType::NameRestrictV<isSlu, isLocal>;
		Slu_DEF_CFG2(DestrNameRestrict, isLocal);
	}
	template<bool isLocal,typename T>
	concept AnyCompoundDestr =
		std::same_as<std::remove_cv_t<T>, DestrPatType::FieldsV<true, isLocal>>
		|| std::same_as<std::remove_cv_t<T>, DestrPatType::ListV<true, isLocal>>;


	template<bool isSlu,bool isLocal>
	using PatV = std::variant<
		PatType::DestrAny,

		PatType::SimpleV<isSlu>,

		PatType::DestrFieldsV<isSlu, isLocal>,
		PatType::DestrListV<isSlu, isLocal>,

		PatType::DestrNameV<isSlu, isLocal>,
		PatType::DestrNameRestrictV<isSlu, isLocal>
	>;
	Slu_DEF_CFG2(Pat, isLocal);

	template<bool isSlu, bool isLocal>
	struct DestrFieldV
	{
		MpItmIdV<isSlu> name;// |(...)| thingy
		PatV<isSlu,isLocal> pat;//May be any type of pattern
	};
	Slu_DEF_CFG2(DestrField,isLocal);
	namespace DestrPatType
	{
		template<bool isSlu, bool isLocal>
		struct FieldsV
		{
			DestrSpecV<isSlu> spec;
			bool extraFields : 1 = false;
			std::vector<DestrFieldV<isSlu, isLocal>> items;
			LocalOrNameV<isSlu, isLocal> name;//May be empty
		};
		template<bool isSlu, bool isLocal>
		struct ListV
		{
			DestrSpecV<isSlu> spec;
			bool extraFields : 1 = false;
			std::vector<PatV<isSlu, isLocal>> items;
			LocalOrNameV<isSlu,isLocal> name;//May be empty
		};

		template<bool isSlu, bool isLocal>
		struct NameV
		{
			LocalOrNameV<isSlu, isLocal> name;
			DestrSpecV<isSlu> spec;
		};
		template<bool isSlu, bool isLocal>
		struct NameRestrictV : NameV<isSlu, isLocal>
		{
			NdPatV<isSlu> restriction;
		};
	}

	template<>
	struct ParameterV<true>
	{
		PatV<true,true> name;
	};
	template<bool isLocal>
	struct ___PatHack : PatV<true, isLocal> {};

	//Common

	namespace SubVarType
	{
		using Deref = std::monostate;

		template<bool isSlu>
		struct NAMEv { MpItmIdV<isSlu> idx; };	// {funcArgs} ‘.’ Name
		Slu_DEF_CFG_CAPS(NAME);

		using parse::ExprV;
		using parse::Expr;
	}

	template<bool isSlu>
	struct SubVarV
	{
		std::vector<ArgFuncCallV<isSlu>> funcCalls;

		std::variant<
			SubVarType::Deref,
			SubVarType::NAMEv<isSlu>,
			SubVarType::ExprV<isSlu>
		> idx;
	};

	Slu_DEF_CFG(SubVar);

	namespace BaseVarType
	{
		using Root = std::monostate;// ":>" // modpath root
		using Local = parse::LocalId;

		template<bool isSlu>
		struct NAMEv
		{
			MpItmIdV<isSlu> v;
		};
		Slu_DEF_CFG_CAPS(NAME);

		using parse::ExprV;
		using parse::Expr;

	}
	template<bool isSlu>
	using BaseVarV = std::variant<
		BaseVarType::Root,
		BaseVarType::Local,
		BaseVarType::NAMEv<isSlu>,
		BaseVarType::ExprV<isSlu>
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
		struct EXPR2EXPRv { parse::ExprV<isSlu> idx; parse::ExprV<isSlu> v; };		// "‘[’ exp ‘]’ ‘=’ exp"

		template<bool isSlu>
		struct NAME2EXPRv { MpItmIdV<isSlu> idx; parse::ExprV<isSlu> v; };	// "Name ‘=’ exp"
	}
	namespace LimPrefixExprType
	{
		template<bool isSlu>
		struct VARv { VarV<isSlu> v; };			// "var"
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
		struct IMPORT { MpItmIdV<true> name; };// use x::y; //name is inside this mp, base is the imported path.
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
		LocalsV<isSlu> local2Mp;
		TyTy type;
		MpItmIdV<isSlu> name;
		ExportData exported = false;
	};
	template<bool isSlu, bool isLocal>
	struct MaybeLocalsV {
		LocalsV<isSlu> local2Mp;
	};
	template<bool isSlu>
	struct MaybeLocalsV<isSlu,true> {};
	template<bool isSlu, bool isLocal>
	struct VarStatBaseV : MaybeLocalsV<isSlu, isLocal>
	{	// "local attnamelist [= explist]" //e.size 0 means "only define, no assign"
		AttribNameListV<isSlu> names;
		ExprListV<isSlu> exprs;
	};
	template<bool isLocal>
	struct VarStatBaseV<true, isLocal> : MaybeLocalsV<true,isLocal>
	{	// "local attnamelist [= explist]" //e.size 0 means "only define, no assign"
		PatV<true, isLocal> names;
		ExprListV<true> exprs;
		ExportData exported = false;
	};

	namespace StatementType
	{
		using Semicol = std::monostate;	// ";"

		template<bool isSlu>
		struct ASSIGNv { std::vector<VarV<isSlu>> vars; ExprListV<isSlu> exprs; };// "varlist = explist" //e.size must be > 0
		Slu_DEF_CFG_CAPS(ASSIGN);

		using parse::FuncCallV;
		using parse::FuncCall;


		template<bool isSlu>
		struct LABELv { MpItmIdV<isSlu> v; };		// "label"
		Slu_DEF_CFG_CAPS(LABEL);
		struct Break {};					// "break"
		template<bool isSlu>
		struct GotoV { MpItmIdV<isSlu> v; };			// "goto Name"
		Slu_DEF_CFG(Goto);

		using parse::BlockV;// "do block end"
		using parse::Block;

		template<bool isSlu>
		struct WhileV { ExprV<isSlu> cond; BlockV<isSlu> bl; };		// "while exp do block end"
		Slu_DEF_CFG(While);

		template<bool isSlu>
		struct RepeatUntilV :WhileV<isSlu> {};						// "repeat block until exp"
		Slu_DEF_CFG(RepeatUntil);

		// "if exp then block {elseif exp then block} [else block] end"
		template<bool isSlu>
		using IfCondV = BaseIfCondV<isSlu, false>;
		Slu_DEF_CFG(IfCond);

		// "for Name = exp , exp [, exp] do block end"
		template<bool isSlu>
		struct ForNumV
		{
			Sel<isSlu, MpItmIdV<isSlu>, PatV<true,true>> varName;
			ExprV<isSlu> start;
			ExprV<isSlu> end;//inclusive
			std::optional<ExprV<isSlu>> step;
			BlockV<isSlu> bl;
		};
		Slu_DEF_CFG(ForNum);

		// "for namelist in explist do block end"
		template<bool isSlu>
		struct ForInV
		{
			Sel<isSlu, NameListV<isSlu>, PatV<true, true>> varNames;
			Sel<isSlu, ExprListV<isSlu>, ExprV<isSlu>> exprs;//size must be > 0
			BlockV<isSlu> bl;
		};
		Slu_DEF_CFG(ForIn);

		template<bool isSlu>
		struct FuncDefBase
		{// "function funcname funcbody"    
			Position place;//Right after func-name
			MpItmIdV<isSlu> name; // name may contain dots, 1 colon if !isSlu
			FunctionV<isSlu> func;
		};
		template<bool isSlu>
		struct FunctionV : FuncDefBase<isSlu> {};
		template<>
		struct FunctionV<true> : FuncDefBase<true>
		{
			ExportData exported = false;
		};
		Slu_DEF_CFG(Function);

		template<bool isSlu>
		struct FnV : FunctionV<isSlu> {};
		Slu_DEF_CFG(Fn);

		template<bool isSlu>
		struct LocalFunctionDefV :FunctionV<isSlu> {};
		Slu_DEF_CFG(LocalFunctionDef);
		// "local function Name funcbody"


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
		using LocalV = VarStatBaseV<isSlu,true>;
		Slu_DEF_CFG(Local);

		// Slu

		template<bool isSlu>
		struct LetV : LocalV<isSlu> {};
		Slu_DEF_CFG(Let);

		template<bool isSlu>
		using ConstV = VarStatBaseV<isSlu, false>;
		Slu_DEF_CFG(Const);

		struct CanonicLocal
		{
			ExprV<true> type;
			LocalId name;
			ExprV<true> value;
			ExportData exported = false;
		};
		struct CanonicGlobal
		{
			ExprV<true> type;
			LocalsV<true> local2Mp;
			MpItmIdV<true> name;
			ExprV<true> value;
			ExportData exported = false;
		};


		template<bool isSlu>
		struct StructV : StructBaseV<ExprV<true>, isSlu> {};
		Slu_DEF_CFG(Struct);

		template<bool isSlu>
		struct UnionV : StructBaseV<TableV<isSlu>, isSlu> {};
		Slu_DEF_CFG(Union);

		template<bool isSlu>
		struct ExternBlockV
		{
			StatListV<isSlu> stats;
			std::string abi;
			Position abiEnd;
			OptSafety safety = OptSafety::DEFAULT;
		};
		Slu_DEF_CFG(ExternBlock);

		template<bool isSlu>
		struct UnsafeBlockV { BlockV<isSlu> bl; };	// "unsafe {...}"
		Slu_DEF_CFG(UnsafeBlock);

		struct UnsafeLabel {};
		struct SafeLabel {};

		struct Use
		{
			MpItmIdV<true> base;//the aliased/imported thing, or modpath base
			UseVariant useVariant;
			ExportData exported = false;
		};

		template<bool isSlu>
		struct DropV
		{
			ExprV<isSlu> expr;
		};
		Slu_DEF_CFG(Drop);

		template<bool isSlu>
		struct ModV
		{
			MpItmIdV<isSlu> name;
			ExportData exported = false;
		};
		Slu_DEF_CFG(Mod);

		template<bool isSlu>
		struct ModAsV : ModV<isSlu>
		{
			BlockV<isSlu> bl;
		};
		Slu_DEF_CFG(ModAs);

	};

	template<bool isSlu>
	using StatementDataV = std::variant <
		StatementType::Semicol,				// ";"

		StatementType::ASSIGNv<isSlu>,			// "varlist = explist"
		StatementType::LocalV<isSlu>,	// "local attnamelist [= explist]"
		StatementType::LetV<isSlu>,	// "let pat [= explist]"
		StatementType::ConstV<isSlu>,	// "const pat [= explist]"
		StatementType::CanonicLocal,
		StatementType::CanonicGlobal,

		StatementType::FuncCallV<isSlu>,		// "functioncall"
		StatementType::LABELv<isSlu>,			// "label"
		StatementType::Break,					// "break"
		StatementType::GotoV<isSlu>,			// "goto Name"
		StatementType::BlockV<isSlu>,			// "do block end"
		StatementType::WhileV<isSlu>,		// "while exp do block end"
		StatementType::RepeatUntilV<isSlu>,	// "repeat block until exp"

		StatementType::IfCondV<isSlu>,	// "if exp then block {elseif exp then block} [else block] end"

		StatementType::ForNumV<isSlu>,	// "for Name = exp , exp [, exp] do block end"
		StatementType::ForInV<isSlu>,	// "for namelist in explist do block end"

		StatementType::LocalFunctionDefV<isSlu>,	// "local function Name funcbody"
		StatementType::FunctionV<isSlu>,		// "function funcname funcbody"
		StatementType::FnV<isSlu>,					// "fn funcname funcbody"

		StatementType::FunctionDeclV<isSlu>,
		StatementType::FnDeclV<isSlu>,

		StatementType::StructV<isSlu>,
		StatementType::UnionV<isSlu>,

		StatementType::ExternBlockV<isSlu>,

		StatementType::UnsafeBlockV<isSlu>,
		StatementType::UnsafeLabel,	// ::: unsafe :
		StatementType::SafeLabel,		// ::: safe :

		StatementType::DropV<isSlu>,	// "drop" Name
		StatementType::Use,				// "use" ...
		StatementType::ModV<isSlu>,		// "mod" Name
		StatementType::ModAsV<isSlu>	// "mod" Name "as" "{" block "}"
	> ;
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
		StatListV<isSlu> code;
		lang::ModPathId mp;
	};
	template<>
	struct ParsedFileV<false>
	{
		BlockV<false> code;
		LocalsV<false> local2Mp;
	};
	Slu_DEF_CFG(ParsedFile);
}