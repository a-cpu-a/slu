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
#include <map>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/ext/CppMatch.hpp>
import slu.lang.basic_state;
#include <slu/lang/Mpc.hpp>
import slu.big_int;
import slu.ast.enums;
#include "SmallEnumList.hpp"
import slu.parse.input;
#include "StateDecls.hpp"

namespace slu::parse
{
	struct ResolvedType;
}

namespace slu::mlvl
{
	bool nearExactCheck(const parse::ResolvedType& subty, const parse::ResolvedType& useTy);
}

namespace slu::parse
{
	inline const size_t TYPE_RES_SIZE_SIZE = 64;//TODO: unhardcode this, allow 8 bits too.
	inline const size_t TYPE_RES_PTR_SIZE = 64;//TODO: unhardcode this, allow 8 bits too.

	struct StructRawType;
	struct UnionRawType;
	struct VariantRawType;
	struct RefChainRawType;
	struct RefSliceRawType;
	struct DelStructRawType
	{
		void operator()(StructRawType* it) const noexcept;
	};
	struct DelUnionRawType
	{
		void operator()(UnionRawType* it) const noexcept;
	};
	struct DelVariantRawType
	{
		void operator()(VariantRawType* it) const noexcept;
	};
	struct DelRefChainRawType
	{
		void operator()(RefChainRawType* it) const noexcept;
	};
	struct DelRefSliceRawType
	{
		void operator()(RefSliceRawType* it) const noexcept;
	};
	namespace RawTypeKind
	{
		using Inferred = std::monostate;
		struct TypeError {};
		using Unresolved = std::unique_ptr<parse::Expr>;

		using String = std::string;

		using Float64 = ExprType::F64;

		using Pos128 = ExprType::P128;
		using Neg128 = ExprType::M128;
		using Uint64 = ExprType::U64;
		using Int64 = ExprType::I64;
	}
	template <class T>
	concept Any128Int = std::same_as<T, parse::RawTypeKind::Pos128>
		|| std::same_as<T, parse::RawTypeKind::Neg128>;
	template <class T>
	concept AnyRawInt = std::same_as<T, parse::RawTypeKind::Uint64>
		|| std::same_as<T, parse::RawTypeKind::Int64>
		|| Any128Int<T>;

	template <bool NEG_MIN, bool NEG_MAX>
	struct Range128
	{
		Integer128<false, NEG_MIN> min;
		Integer128<false, NEG_MAX> max;

		template<AnyRawInt T>
		constexpr bool isInside(const T v) const {
			return v >= min && v <= max;
		}
		constexpr bool isOnly(const AnyRawInt auto o) const {
			return min == o && max == o;
		}
		constexpr auto operator<=>(const Range128&) const = default;
	};

	namespace RawTypeKind
	{
		using Range128Pp = Range128<false, false>;
		using Range128Np = Range128<true, false>;
		using Range128Nn = Range128<true, true>;

		struct Range64
		{
			Int64 min;
			Int64 max;
			template<AnyRawInt T>
			constexpr bool isInside(const T v) const {
				if constexpr (std::same_as<T, Uint64>)
				{
					if (v > (uint64_t)INT64_MAX) return false;
					return isInside((Int64)v);
				}
				else
					return v >= min && v <= max;
			}
			template<AnyRawInt T>
			constexpr bool isOnly(const T o) const {
				if constexpr (std::same_as<T, Uint64>)
				{
					if (o > (uint64_t)INT64_MAX) return false;
					return isOnly((Int64)o);
				}
				return min == o && max == o;
			}
			constexpr auto operator<=>(const Range64&) const = default;
		};
		using Variant = std::unique_ptr<VariantRawType, DelVariantRawType>;
		using Union = std::unique_ptr<UnionRawType, DelUnionRawType>;
		using Struct = std::unique_ptr<StructRawType, DelStructRawType>;
		using RefChain = std::unique_ptr<RefChainRawType, DelRefChainRawType>;
		using RefSlice = std::unique_ptr<RefSliceRawType, DelRefSliceRawType>;
	}
	using RawType = std::variant <
		RawTypeKind::TypeError,
		RawTypeKind::Inferred,
		RawTypeKind::Unresolved,

		RawTypeKind::String,
		RawTypeKind::Float64,
		RawTypeKind::Pos128,
		RawTypeKind::Neg128,
		RawTypeKind::Range128Pp,
		RawTypeKind::Range128Np,
		RawTypeKind::Range128Nn,
		RawTypeKind::Int64,
		RawTypeKind::Uint64,
		RawTypeKind::Range64,
		RawTypeKind::Variant,
		RawTypeKind::Union,
		RawTypeKind::Struct,
		RawTypeKind::RefChain,
		RawTypeKind::RefSlice
	>;

	template <class T>
	concept AnyRawRange = std::same_as<T, parse::RawTypeKind::Range64>
		|| std::same_as<T, parse::RawTypeKind::Range128Nn>
		|| std::same_as<T, parse::RawTypeKind::Range128Pp>
		|| std::same_as<T, parse::RawTypeKind::Range128Np>;
	using Range129 = std::variant<
		parse::RawTypeKind::Range128Pp,
		parse::RawTypeKind::Range128Np,
		parse::RawTypeKind::Range128Nn
	>;

	constexpr uint64_t abs(int64_t v)
	{
		if (v == INT64_MIN)
			return INT64_MAX + 1ULL;
		return v < 0 ? -v : v;
	}
	constexpr auto r129Get(const Range129& v, auto&& visitor) {
		return std::visit(std::move(visitor), v);
	}
	template <class MinT, class MaxT>
	constexpr auto r129From(const MinT& min, const MaxT& max) {
		constexpr bool minP = std::same_as<MinT, Integer128<false, false>>;
		constexpr bool maxP = std::same_as<MaxT, Integer128<false, false>>;

		static_assert(!(minP && !maxP), "Found positive min, and negative max");

		if constexpr (minP && maxP)
			return RawTypeKind::Range128Pp{ .min = min, .max = max };
		else if constexpr (!minP && maxP)
			return RawTypeKind::Range128Np{ .min = min, .max = max };
		else
			return RawTypeKind::Range128Nn{ .min = min, .max = max };
	}
	constexpr Range129 range129FromInt(uint64_t v) {
		return RawTypeKind::Range128Pp{ .min = v, .max = v };
	}
	constexpr Range129 range129FromInt(int64_t v) {
		if (v < 0)
			return RawTypeKind::Range128Nn{ .min = abs(v), .max = abs(v) };
		return range129FromInt((uint64_t)v);
	}
	constexpr Range129 range129FromInt(Any128Int auto v)
	{
		if constexpr (std::same_as<decltype(v), RawTypeKind::Neg128>)
			return RawTypeKind::Range128Nn{ .min = v, .max = v };
		else
			return RawTypeKind::Range128Pp{ .min = v, .max = v };
	}
	constexpr Range129 range129From64(const RawTypeKind::Range64 v)
	{
		if (v.min < 0)
		{
			if (v.max < 0)
				return RawTypeKind::Range128Nn{ .min = abs(v.min), .max = abs(v.max) };
			return RawTypeKind::Range128Np{ .min = abs(v.min), .max = (uint64_t)v.max };
		}
		_ASSERT(v.max >= 0);
		return RawTypeKind::Range128Pp{ .min = (uint64_t)v.min, .max = (uint64_t)v.max };
	}
	constexpr size_t calcRangeBits(const RawTypeKind::Range64 range)
	{
		uint64_t vals;//implicit +1.
		if (range.min < 0)
		{
			vals = abs(range.min);
			if (range.max < 0)
				vals -= abs(range.max);
			else
				vals += range.max;
		}
		else
			vals = range.max - range.min;
		return std::bit_width(vals);
	}
	template<AnyRawRange T>
	constexpr size_t calcRangeBits(const T& range)
	{
		constexpr bool minP = std::same_as<T, RawTypeKind::Range128Pp>;
		constexpr bool maxP = std::same_as<T, RawTypeKind::Range128Np>
			|| std::same_as<T, RawTypeKind::Range128Pp>;
		static_assert(!(minP && !maxP), "Found positive min, and negative max");

		if constexpr (minP && maxP)
			return (range.max - range.min).bitWidth();
		else if constexpr (!minP && maxP)
			return (range.min.abs() + range.max).bitWidth();
		else
			return (range.min.abs() - range.max.abs()).bitWidth();
	}

	template <class T>
	concept AnyRawIntOrRange = AnyRawInt<T> || AnyRawRange<T>;

	RawType cloneRawType(const RawType& t);

	constexpr uint8_t alignDataFromSize(size_t bits)
	{
		if (bits == 0)
			return 0;
		if (bits <= 128)
			return (uint8_t)std::bit_width(bits-1);
		return 7;
	}

	struct ResolvedType
	{
		constexpr static size_t INCOMPLETE_MARK = (1ULL << 50) - 1;
		constexpr static size_t UNSIZED_MARK = INCOMPLETE_MARK - 1;

		RawType base;
		size_t size : 46;//in bits. element type size, ignoring outerSliceDims.
		size_t alignmentData : 4 = 0;//log2(alignment in bits). 1bit .. 4096bytes
		size_t outerSliceDims : 13 = 0;
		size_t hasMut : 1 = false;

		ResolvedType clone() const {
			return { .base = cloneRawType(base),
				.size = size,
				.alignmentData = alignmentData,
				.outerSliceDims = outerSliceDims,
				.hasMut = hasMut 
			};
		}
		constexpr size_t alignBits() const {
			return 2ULL << alignmentData;
		}
		constexpr bool isComplete() const {
			return size != INCOMPLETE_MARK;
		}
		constexpr bool isSized() const {
			return outerSliceDims == 0 && size != UNSIZED_MARK;
		}
		constexpr std::optional<lang::MpItmId> getStructName() const;
		constexpr bool isBool() const
		{
			if (size > 1) return false;
			auto optName = getStructName();
			if (!optName.has_value()) return false;

			MpItmId name = *optName;
			return name == mpc::STD_BOOL
				|| name == mpc::STD_BOOL_FALSE
				|| name == mpc::STD_BOOL_TRUE;
		}

		static ResolvedType getInferred() {
			return { .base = parse::RawTypeKind::Inferred{},.size = INCOMPLETE_MARK };
		}
		static ResolvedType newError() {
			return { .base = parse::RawTypeKind::TypeError{},.size = INCOMPLETE_MARK };
		}

		static ResolvedType getConstType(RawType&& v) {
			return { .base = std::move(v),.size = 0,.alignmentData=0 /*Known value, not stored*/ };
		}
		static ResolvedType newU8() {
			return { .base = RawTypeKind::Range64{.min=0,.max=UINT8_MAX},.size = 8,.alignmentData=alignDataFromSize(8)};
		}
		static ResolvedType newIntRange(const auto& range) {
			const bool minIsI64 = range.min <= INT64_MAX && range.min >= INT64_MIN;
			if (range.min == range.max)
			{
				if (minIsI64)
					return ResolvedType::getConstType(RawTypeKind::Int64(range.min.lo));
				if (range.min >= 0ULL && range.min <= UINT64_MAX)
					return ResolvedType::getConstType(RawTypeKind::Uint64(range.min.lo));
				return ResolvedType::getConstType(RawType(range.min));
			}
			if (minIsI64 && range.max <= INT64_MAX && range.max >= INT64_MIN)
			{
				ResolvedType res = { .base = RawTypeKind::Range64{
				(int64_t)range.min.lo,
				(int64_t)range.max.lo},
				.size = calcRangeBits(range)
				};
				res.alignmentData = alignDataFromSize(res.size);
				return res;
			}
			ResolvedType res = { .base = range,.size = calcRangeBits(range) };
			res.alignmentData = alignDataFromSize(res.size);
			return res;
		}
	};
	struct VariantRawType
	{
		std::vector<ResolvedType> options;

		static RawTypeKind::Variant newRawTy() {
			auto& elems = *(new VariantRawType());
			return RawTypeKind::Variant{ &elems };
		}
		RawTypeKind::Variant cloneRaw() const
		{
			RawTypeKind::Variant res = newRawTy();
			res->options.reserve(options.size());
			for (const auto& i : options)
				res->options.emplace_back(i.clone());
			return res;
		}

		constexpr std::pair<size_t, uint8_t> calcSizeAndAlign() const {
			size_t size = 0;
			uint8_t align = 0;
			for (const auto& i : options)
			{
				align = std::max(align, (uint8_t)i.alignmentData);
				if (!i.isComplete())
				{
					size = ResolvedType::INCOMPLETE_MARK;
					break;
				}
				else if (!i.isSized())
				{
					size = ResolvedType::UNSIZED_MARK;
					continue;
				}
				if (i.size > size)
					size = i.size;
			}
			if (!options.empty() && size != ResolvedType::INCOMPLETE_MARK && size != ResolvedType::UNSIZED_MARK)
			{
				//size = (size + 7) & (~0b111);// round up to 8 bits
				size += std::bit_width(options.size() - 1);//add bits for the index
			}
			align = std::max(align, alignDataFromSize(size));
			return {size, align};
		}

		template<std::same_as<ResolvedType>... Ts>
		static ResolvedType newTy(Ts&&... t) {
			auto& elems = *(new VariantRawType());
			((elems.options.emplace_back(std::move(t))), ...);
			auto [sz, alignData] = elems.calcSizeAndAlign();
			return { .base = parse::RawTypeKind::Variant{&elems},.size = sz,.alignmentData = alignData};
		}
		bool nearlyExact(const VariantRawType& o) const {
			if (options.size() != o.options.size()) return false;
			for (size_t i = 0; i < options.size(); ++i)
			{
				if (!mlvl::nearExactCheck(options[i], o.options[i]))
					return false;
			}
			return true;
		}
	};
	struct StructRawType
	{
		std::vector<ResolvedType> fields;
		std::vector<std::string> fieldNames;//may be hex ints, like "0x1"
		std::vector<size_t> fieldOffsets;//Only defined for fields that have a size>0, also its in bits.
		lang::MpItmId name;//if empty, then structural / table / tuple / array
		//~StructRawType() {
		//	fields.~vector();
		//	fieldNames.~vector();
		//	fieldOffsets.~vector();
		//}

		static RawTypeKind::Struct newRawTy() {
			auto& elems = *(new StructRawType());
			return RawTypeKind::Struct{ &elems };
		}
		RawTypeKind::Struct cloneRaw() const
		{
			RawTypeKind::Struct res = newRawTy();
			res->fields.reserve(fields.size());
			for (const auto& i : fields)
				res->fields.emplace_back(i.clone());
			res->fieldNames = fieldNames;
			res->fieldOffsets = fieldOffsets;
			res->name = name;
			return res;
		}

		static parse::RawTypeKind::Struct newNamed(lang::MpItmId name) {
			auto t = newRawTy();
			t->name = name;
			return t;
		}
		static ResolvedType newZstTy(lang::MpItmId name) {
			return ResolvedType::getConstType(newNamed(name));
		}
		static parse::RawTypeKind::Struct boolStruct() {
			RawTypeKind::Struct thing = newRawTy();
			thing->fields.emplace_back(VariantRawType::newTy(
				newFalseTy(),
				newTrueTy()
			));
			thing->fieldNames.push_back("0x1");
			thing->name = mpc::STD_BOOL;
			return thing;
		}
		static ResolvedType newTrueTy() {
			return newZstTy(mpc::STD_BOOL_TRUE);
		}
		static ResolvedType newFalseTy() {
			return newZstTy(mpc::STD_BOOL_FALSE);
		}
		static ResolvedType newBoolTy(const bool tr, const bool fa) {
			if (tr && fa)
				return { .base = parse::RawTypeKind::Struct{boolStruct()},.size = 1,.alignmentData=alignDataFromSize(1)};
			if (tr) return newTrueTy();
			_ASSERT(fa);
			return newFalseTy();
		}
		bool nearlyExact(const StructRawType& o) const {
			if (fields.size() != o.fields.size()) return false;
			if (name != o.name) return false;
			for (size_t i = 0; i < fields.size(); ++i)
			{
				if (fieldNames[i] != o.fieldNames[i])
					return false;
				if (!mlvl::nearExactCheck(fields[i], o.fields[i]))
					return false;
			}
			return true;
		}
	};
	struct UnionRawType
	{
		std::vector<ResolvedType> fields;
		std::vector<std::string> fieldNames;//may be hex ints, like "0x1"
		lang::MpItmId name;//if empty, then structural / table / tuple / array

		static RawTypeKind::Union newRawTy() {
			auto& elems = *(new UnionRawType());
			return RawTypeKind::Union{ &elems };
		}
		RawTypeKind::Union cloneRaw() const
		{
			RawTypeKind::Union res = newRawTy();
			res->fields.reserve(fields.size());
			for (const auto& i : fields)
				res->fields.emplace_back(i.clone());
			res->fieldNames = fieldNames;
			res->name = name;
			return res;
		}
		bool nearlyExact(const UnionRawType& o) const {
			if (fields.size() != o.fields.size()) return false;
			if (name != o.name) return false;
			for (size_t i = 0; i < fields.size(); ++i)
			{
				if (fieldNames[i] != o.fieldNames[i])
					return false;
				if (!mlvl::nearExactCheck(fields[i], o.fields[i]))
					return false;
			}
			return true;
		}
	};
	struct RefSigil
	{
		lang::MpItmId life;
		ast::UnOpType refType;

		constexpr auto operator<=>(const RefSigil&) const = default;
	};
	struct RefChainRawType
	{
		ResolvedType elem;
		std::vector<RefSigil> chain;//In application order, so {&share,&mut} -> &mut &share T

		static ResolvedType newPtrTy(parse::ResolvedType&& t) {
			auto& val = *(new RefChainRawType(std::move(t), { RefSigil{.refType=ast::UnOpType::TO_PTR} }));
			return { .base = parse::RawTypeKind::RefChain{&val},.size = TYPE_RES_PTR_SIZE, .alignmentData=alignDataFromSize(TYPE_RES_PTR_SIZE)};
		}
		static RawTypeKind::RefChain newRawTy() {
			auto& elems = *(new RefChainRawType());
			return RawTypeKind::RefChain{ &elems };
		}
		RawTypeKind::RefChain cloneRaw() const
		{
			RawTypeKind::RefChain res = newRawTy();
			res->elem = elem.clone();
			res->chain = chain;
			return res;
		}
	};
	struct RefSliceRawType
	{
		ResolvedType elem;//dims stored in here.
		ast::UnOpType refType;

		static RawTypeKind::RefSlice newRawTy() {
			auto& elems = *(new RefSliceRawType());
			return RawTypeKind::RefSlice{ &elems };
		}
		RawTypeKind::RefSlice cloneRaw() const
		{
			RawTypeKind::RefSlice res = newRawTy();
			res->elem = elem.clone();
			res->refType = refType;
			return res;
		}
	};
	void ::slu::parse::DelStructRawType::operator()(StructRawType* it) const noexcept {
		delete it;
	}
	void ::slu::parse::DelUnionRawType::operator()(UnionRawType* it) const noexcept {
		delete it;
	}
	void ::slu::parse::DelVariantRawType::operator()(VariantRawType* it) const noexcept {
		delete it;
	}
	void ::slu::parse::DelRefChainRawType::operator()(RefChainRawType* it) const noexcept {
		delete it;
	}
	void ::slu::parse::DelRefSliceRawType::operator()(RefSliceRawType* it) const noexcept {
		delete it;
	}

	constexpr std::optional<lang::MpItmId> ResolvedType::getStructName() const
	{
		if (outerSliceDims != 0) return std::nullopt;
		if (!std::holds_alternative<RawTypeKind::Struct>(base))
			return std::nullopt;
		return std::get<RawTypeKind::Struct>(base)->name;
	}
	inline RawType cloneRawType(const RawType& t)
	{
		return ezmatch(t)(
			varcase(const auto&) { return RawType{ var }; },
			varcase(const RawTypeKind::Unresolved&)->RawType {
			throw std::runtime_error("Cannot clone unresolved type");
		},
			varcase(const RawTypeKind::Struct&) {
			return RawType{ var->cloneRaw() };
		},
			varcase(const RawTypeKind::Union&) {
			return RawType{ var->cloneRaw() };
		},
			varcase(const RawTypeKind::Variant&) {
			return RawType{ var->cloneRaw() };
		},
			varcase(const RawTypeKind::RefChain&) {
			return RawType{ var->cloneRaw() };
		},
			varcase(const RawTypeKind::RefSlice&) {
			return RawType{ var->cloneRaw() };
		}

			);
	}
}