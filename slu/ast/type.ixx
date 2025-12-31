module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <bit>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <slu/ext/CppMatch.hpp>
#include <slu/Panic.hpp>
export module slu.ast.type;
import slu.big_int;
import slu.ast.enums;
import slu.ast.small_enum_list;
import slu.ast.state_decls;
import slu.lang.basic_state;
import slu.lang.mpc;
import slu.parse.input;

namespace slu::parse //TODO: ast
{
	extern "C++"
	{
	export struct ResolvedType;
	}
} //namespace slu::parse

namespace slu::mlvl
{
	extern "C++"
	{
	bool nearExactCheck(
	    const parse::ResolvedType& subty, const parse::ResolvedType& useTy);
	}
} //namespace slu::mlvl

namespace slu::parse //TODO: ast
{
	export constexpr size_t TYPE_RES_SIZE_SIZE
	    = 64; //TODO: unhardcode this, allow 8 bits too.
	export constexpr size_t TYPE_RES_PTR_SIZE
	    = 64; //TODO: unhardcode this, allow 8 bits too.

	export struct StructRawType;
	export struct UnionRawType;
	export struct VariantRawType;
	export struct RefRawType;
	export struct PtrRawType;
	export struct SliceRawType;
	export struct DelStructRawType
	{
		void operator()(StructRawType* it) const noexcept;
	};
	export struct DelUnionRawType
	{
		void operator()(UnionRawType* it) const noexcept;
	};
	export struct DelVariantRawType
	{
		void operator()(VariantRawType* it) const noexcept;
	};
	export struct DelRefRawType
	{
		void operator()(RefRawType* it) const noexcept;
	};
	export struct DelPtrRawType
	{
		void operator()(PtrRawType* it) const noexcept;
	};
	export struct DelSliceRawType
	{
		void operator()(SliceRawType* it) const noexcept;
	};
	extern "C++"
	{
	export struct DelExpr
	{
		void operator()(parse::Expr* it) const noexcept;
	};
	}
	namespace RawTypeKind
	{
		export using Inferred = std::monostate;
		export struct TypeError
		{};
		export using Unresolved = std::unique_ptr<parse::Expr, DelExpr>;

		export using String = std::string;

		export using Float64 = ExprType::F64;

		export using Pos128 = ExprType::P128;
		export using Neg128 = ExprType::M128;
		export using Uint64 = ExprType::U64;
		export using Int64 = ExprType::I64;
	} //namespace RawTypeKind
	export template<class T>
	concept Any128Int = std::same_as<T, parse::RawTypeKind::Pos128>
	    || std::same_as<T, parse::RawTypeKind::Neg128>;
	export template<class T>
	concept AnyRawInt = std::same_as<T, parse::RawTypeKind::Uint64>
	    || std::same_as<T, parse::RawTypeKind::Int64> || Any128Int<T>;

	export template<bool NEG_MIN, bool NEG_MAX> struct Range128
	{
		Integer128<false, NEG_MIN> min;
		Integer128<false, NEG_MAX> max;

		template<AnyRawInt T> constexpr bool isInside(const T v) const
		{
			return v >= min && v <= max;
		}
		constexpr bool isOnly(const AnyRawInt auto o) const
		{
			return min == o && max == o;
		}
		constexpr auto operator<=>(const Range128&) const = default;
	};

	namespace RawTypeKind
	{
		export using Range128Pp = Range128<false, false>;
		export using Range128Np = Range128<true, false>;
		export using Range128Nn = Range128<true, true>;

		export struct Range64
		{
			Int64 min;
			Int64 max;
			template<AnyRawInt T> constexpr bool isInside(const T v) const
			{
				if constexpr (std::same_as<T, Uint64>)
				{
					if (v > (uint64_t)INT64_MAX)
						return false;
					return isInside((Int64)v);
				} else
					return v >= min && v <= max;
			}
			template<AnyRawInt T> constexpr bool isOnly(const T o) const
			{
				if constexpr (std::same_as<T, Uint64>)
				{
					if (o > (uint64_t)INT64_MAX)
						return false;
					return isOnly((Int64)o);
				}
				return min == o && max == o;
			}
			constexpr auto operator<=>(const Range64&) const = default;
		};
		export using Variant
		    = std::unique_ptr<VariantRawType, DelVariantRawType>;
		export using Union = std::unique_ptr<UnionRawType, DelUnionRawType>;
		export using Struct = std::unique_ptr<StructRawType, DelStructRawType>;
		export using Ref = std::unique_ptr<RefRawType, DelRefRawType>;
		export using Ptr = std::unique_ptr<PtrRawType, DelPtrRawType>;
		export using Slice = std::unique_ptr<SliceRawType, DelSliceRawType>;
	} //namespace RawTypeKind
	export using RawType = std::variant<RawTypeKind::TypeError,
	    RawTypeKind::Inferred, RawTypeKind::Unresolved,

	    RawTypeKind::String, RawTypeKind::Float64, RawTypeKind::Pos128,
	    RawTypeKind::Neg128, RawTypeKind::Range128Pp, RawTypeKind::Range128Np,
	    RawTypeKind::Range128Nn, RawTypeKind::Int64, RawTypeKind::Uint64,
	    RawTypeKind::Range64, RawTypeKind::Variant, RawTypeKind::Union,
	    RawTypeKind::Struct, RawTypeKind::Ref, RawTypeKind::Ptr,
	    RawTypeKind::Slice>;

	export template<class T>
	concept AnyRawRange = std::same_as<T, parse::RawTypeKind::Range64>
	    || std::same_as<T, parse::RawTypeKind::Range128Nn>
	    || std::same_as<T, parse::RawTypeKind::Range128Pp>
	    || std::same_as<T, parse::RawTypeKind::Range128Np>;
	export using Range129 = std::variant<parse::RawTypeKind::Range128Pp,
	    parse::RawTypeKind::Range128Np, parse::RawTypeKind::Range128Nn>;

	export constexpr uint64_t abs(int64_t v)
	{
		if (v == INT64_MIN)
			return INT64_MAX + 1ULL;
		return v < 0 ? -v : v;
	}
	export constexpr auto r129Get(const Range129& v, auto&& visitor)
	{
		return std::visit(std::move(visitor), v);
	}
	export template<class MinT, class MaxT>
	constexpr auto r129From(const MinT& min, const MaxT& max)
	{
		constexpr bool minP = std::same_as<MinT, Integer128<false, false>>;
		constexpr bool maxP = std::same_as<MaxT, Integer128<false, false>>;

		static_assert(!(minP && !maxP), "Found positive min, and negative max");

		if constexpr (minP && maxP)
			return RawTypeKind::Range128Pp{.min = min, .max = max};
		else if constexpr (!minP && maxP)
			return RawTypeKind::Range128Np{.min = min, .max = max};
		else
			return RawTypeKind::Range128Nn{.min = min, .max = max};
	}
	export constexpr Range129 range129FromInt(uint64_t v)
	{
		return RawTypeKind::Range128Pp{.min = v, .max = v};
	}
	export constexpr Range129 range129FromInt(int64_t v)
	{
		if (v < 0)
			return RawTypeKind::Range128Nn{.min = abs(v), .max = abs(v)};
		return range129FromInt((uint64_t)v);
	}
	export constexpr Range129 range129FromInt(Any128Int auto v)
	{
		if constexpr (std::same_as<decltype(v), RawTypeKind::Neg128>)
			return RawTypeKind::Range128Nn{.min = v, .max = v};
		else
			return RawTypeKind::Range128Pp{.min = v, .max = v};
	}
	export constexpr Range129 range129From64(const RawTypeKind::Range64 v)
	{
		if (v.min < 0)
		{
			if (v.max < 0)
				return RawTypeKind::Range128Nn{
				    .min = abs(v.min), .max = abs(v.max)};
			return RawTypeKind::Range128Np{
			    .min = abs(v.min), .max = (uint64_t)v.max};
		}
		Slu_assertOp(v.max, >=, 0);
		return RawTypeKind::Range128Pp{
		    .min = (uint64_t)v.min, .max = (uint64_t)v.max};
	}
	export constexpr size_t calcRangeBits(const RawTypeKind::Range64 range)
	{
		uint64_t vals; //implicit +1.
		if (range.min < 0)
		{
			vals = abs(range.min);
			if (range.max < 0)
				vals -= abs(range.max);
			else
				vals += range.max;
		} else
			vals = range.max - range.min;
		return std::bit_width(vals);
	}
	export template<AnyRawRange T>
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

	export template<class T>
	concept AnyRawIntOrRange = AnyRawInt<T> || AnyRawRange<T>;

	export RawType cloneRawType(const RawType& t);

	export constexpr uint8_t alignDataFromSize(size_t bits)
	{
		if (bits == 0)
			return 0;
		if (bits <= 128)
			return (uint8_t)std::bit_width(bits - 1);
		return 7;
	}

	extern "C++"
	{
	export struct ResolvedType
	{
		constexpr static size_t INCOMPLETE_MARK
		    = (1ULL << 46) - 1; // Means alignment & size unknown.
		constexpr static size_t UNSIZED_MARK = INCOMPLETE_MARK - 1;
		constexpr static size_t COMPUTED_ALIGN_MARK
		    = (1 << 5) - 1; // If alignmentData is this value, then alignment is
		                    // computed at runtime.

		RawType base;
		size_t size          : 46; //in bits. element type size
		size_t alignmentData : 5
		    = 0; // log2(alignment in bits). 1bit .. 134217728bytes =128MiB.
		         // //TODO: more powerful system allowing for multi-aligned
		         // types, cooler sub-byte alignments.
		size_t hasMut    : 1 = false;
		size_t _reserved : (8 + 4) = 0;

		ResolvedType clone() const
		{
			return {.base = cloneRawType(base),
			    .size = size,
			    .alignmentData = alignmentData,
			    .hasMut = hasMut};
		}
		constexpr size_t alignBits() const
		{
			return 2ULL << alignmentData;
		}
		constexpr bool isComplete() const
		{
			return size != INCOMPLETE_MARK;
		}
		constexpr bool isSized() const
		{
			return size != UNSIZED_MARK;
		}
		constexpr std::optional<lang::MpItmId> getStructName() const;
		constexpr bool isBool() const
		{
			if (size > 1)
				return false;
			auto optName = getStructName();
			if (!optName.has_value())
				return false;

			lang::MpItmId name = *optName;
			return name == mpc::STD_BOOL || name == mpc::STD_BOOL_FALSE
			    || name == mpc::STD_BOOL_TRUE;
		}

		static ResolvedType getInferred()
		{
			return {.base = parse::RawTypeKind::Inferred{},
			    .size = INCOMPLETE_MARK};
		}
		static ResolvedType newError()
		{
			return {.base = parse::RawTypeKind::TypeError{},
			    .size = INCOMPLETE_MARK};
		}

		static ResolvedType getConstType(RawType&& v)
		{
			return {.base = std::move(v),
			    .size = 0,
			    .alignmentData = 0 /*Known value, not stored*/};
		}
		static ResolvedType newU8()
		{
			return {
			    .base = RawTypeKind::Range64{.min = 0, .max = UINT8_MAX},
			    .size = 8,
			    .alignmentData = alignDataFromSize(8)
            };
		}
		static ResolvedType newUtf8Unit()
		{
			return {
			    .base = RawTypeKind::Range64{.min = 0, .max = 0xF4},
			    .size = 8,
			    .alignmentData = alignDataFromSize(8)
            };
		}
		static ResolvedType newIntRange(const auto& range)
		{
			const bool minIsI64
			    = range.min <= INT64_MAX && range.min >= INT64_MIN;
			if (range.min == range.max)
			{
				if (minIsI64)
					return ResolvedType::getConstType(
					    RawTypeKind::Int64(range.min.lo));
				if (range.min >= 0ULL && range.min <= UINT64_MAX)
					return ResolvedType::getConstType(
					    RawTypeKind::Uint64(range.min.lo));
				return ResolvedType::getConstType(RawType(range.min));
			}
			if (minIsI64 && range.max <= INT64_MAX && range.max >= INT64_MIN)
			{
				ResolvedType res = {
				    .base = RawTypeKind::Range64{(int64_t)range.min.lo,
				                                 (int64_t)range.max.lo},
				    .size = calcRangeBits(range)
                };
				res.alignmentData = alignDataFromSize(res.size);
				return res;
			}
			ResolvedType res = {.base = range, .size = calcRangeBits(range)};
			res.alignmentData = alignDataFromSize(res.size);
			return res;
		}
	};
	}
	export struct VariantRawType
	{
		std::vector<ResolvedType> options;

		static RawTypeKind::Variant newRawTy()
		{
			auto& elems = *(new VariantRawType());
			return RawTypeKind::Variant{&elems};
		}
		RawTypeKind::Variant cloneRaw() const
		{
			RawTypeKind::Variant res = newRawTy();
			res->options.reserve(options.size());
			for (const auto& i : options)
				res->options.emplace_back(i.clone());
			return res;
		}

		constexpr std::pair<size_t, uint8_t> calcSizeAndAlign() const
		{
			size_t size = 0;
			uint8_t align = 0;
			for (const auto& i : options)
			{
				align = std::max(align, (uint8_t)i.alignmentData);
				if (!i.isComplete())
				{
					size = ResolvedType::INCOMPLETE_MARK;
					break;
				} else if (!i.isSized())
				{
					size = ResolvedType::UNSIZED_MARK;
					continue;
				}
				if (i.size > size)
					size = i.size;
			}
			if (!options.empty() && size != ResolvedType::INCOMPLETE_MARK
			    && size != ResolvedType::UNSIZED_MARK)
			{
				//size = (size + 7) & (~0b111);// round up to 8 bits
				size += std::bit_width(
				    options.size() - 1); //add bits for the index
			}
			align = std::max(align, alignDataFromSize(size));
			return {size, align};
		}

		template<std::same_as<ResolvedType>... Ts>
		static ResolvedType newTy(Ts&&... t)
		{
			auto& elems = *(new VariantRawType());
			((elems.options.emplace_back(std::move(t))), ...);
			auto [sz, alignData] = elems.calcSizeAndAlign();
			return {.base = parse::RawTypeKind::Variant{&elems},
			    .size = sz,
			    .alignmentData = alignData};
		}
		bool nearlyExact(const VariantRawType& o) const
		{
			if (options.size() != o.options.size())
				return false;
			for (size_t i = 0; i < options.size(); ++i)
			{
				if (!mlvl::nearExactCheck(options[i], o.options[i]))
					return false;
			}
			return true;
		}
	};
	export struct SliceRawType
	{
		ResolvedType elem;

		static RawTypeKind::Slice newRawTy()
		{
			auto& elems = *(new SliceRawType());
			return RawTypeKind::Slice{&elems};
		}
		static ResolvedType newTy(parse::ResolvedType&& elem)
		{
			size_t sz;
			if (!elem.isComplete())
				sz = ResolvedType::INCOMPLETE_MARK;
			else if (elem.size == 0)
				sz = TYPE_RES_SIZE_SIZE; //zst elements dont need a stride
			else
				sz = ResolvedType::UNSIZED_MARK;

			auto& val = *(new SliceRawType(std::move(elem)));
			return {.base = parse::RawTypeKind::Slice{&val},
			    .size = sz,
			    .alignmentData = std::max((uint8_t)elem.alignmentData,
			        alignDataFromSize(TYPE_RES_SIZE_SIZE))};
		}
		RawTypeKind::Slice cloneRaw() const
		{
			RawTypeKind::Slice res = newRawTy();
			res->elem = elem.clone();
			return res;
		}
	};
	export struct StructRawType
	{
		std::vector<ResolvedType> fields;
		std::vector<std::string> fieldNames; //may be hex ints, like "0x1"
		std::vector<size_t> fieldOffsets; //Only defined for fields that have a
		                                  //size>0, also its in bits.
		lang::MpItmId name; //if empty, then structural / table / tuple / array

		static RawTypeKind::Struct newRawTy()
		{
			auto& elems = *(new StructRawType());
			return RawTypeKind::Struct{&elems};
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

		static parse::RawTypeKind::Struct newNamed(lang::MpItmId name)
		{
			auto t = newRawTy();
			t->name = name;
			return t;
		}
		static ResolvedType newZstTy(lang::MpItmId name)
		{
			return ResolvedType::getConstType(newNamed(name));
		}
		static parse::RawTypeKind::Struct boolStruct()
		{
			RawTypeKind::Struct thing = newRawTy();
			thing->fields.emplace_back(
			    VariantRawType::newTy(newFalseTy(), newTrueTy()));
			thing->fieldNames.push_back("0x1");
			thing->name = mpc::STD_BOOL;
			return thing;
		}
		static ResolvedType newTrueTy()
		{
			return newZstTy(mpc::STD_BOOL_TRUE);
		}
		static ResolvedType newFalseTy()
		{
			return newZstTy(mpc::STD_BOOL_FALSE);
		}
		static ResolvedType newBoolTy(const bool tr, const bool fa)
		{
			if (tr && fa)
				return {.base = parse::RawTypeKind::Struct{boolStruct()},
				    .size = 1,
				    .alignmentData = alignDataFromSize(1)};
			if (tr)
				return newTrueTy();
			Slu_assert(fa);
			return newFalseTy();
		}
		static parse::RawTypeKind::Struct strStruct()
		{
			RawTypeKind::Struct thing = newRawTy();
			thing->fields.emplace_back(
			    SliceRawType::newTy(ResolvedType::newUtf8Unit()));
			thing->fieldNames.push_back("0x1");
			thing->name = mpc::STD_STR;
			return thing;
		}
		static ResolvedType newStrTy()
		{
			return {.base = parse::RawTypeKind::Struct{strStruct()},
			    .size = ResolvedType::UNSIZED_MARK,
			    .alignmentData = alignDataFromSize(TYPE_RES_SIZE_SIZE)};
		}
		bool nearlyExact(const StructRawType& o) const
		{
			if (fields.size() != o.fields.size())
				return false;
			if (name != o.name)
				return false;
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
	export struct UnionRawType
	{
		std::vector<ResolvedType> fields;
		std::vector<std::string> fieldNames; //may be hex ints, like "0x1"
		lang::MpItmId name; //if empty, then structural / table / tuple / array

		static RawTypeKind::Union newRawTy()
		{
			auto& elems = *(new UnionRawType());
			return RawTypeKind::Union{&elems};
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
		bool nearlyExact(const UnionRawType& o) const
		{
			if (fields.size() != o.fields.size())
				return false;
			if (name != o.name)
				return false;
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
	export struct RefRawType
	{
		ResolvedType elem;
		lang::MpItmId life;
		ast::UnOpType refType;

		static RawTypeKind::Ref newRawTy()
		{
			auto& elems = *(new RefRawType());
			return RawTypeKind::Ref{&elems};
		}
		RawTypeKind::Ref cloneRaw() const
		{
			RawTypeKind::Ref res = newRawTy();
			res->elem = elem.clone();
			res->life = life;
			res->refType = refType;
			return res;
		}
	};
	export struct PtrRawType
	{
		ResolvedType elem;
		ast::UnOpType ptrType;

		static ResolvedType newTy(
		    parse::ResolvedType&& t, ast::UnOpType ptrType)
		{
			auto& val = *(new PtrRawType(std::move(t), ptrType));
			return {.base = parse::RawTypeKind::Ptr{&val},
			    .size = TYPE_RES_PTR_SIZE,
			    .alignmentData = alignDataFromSize(TYPE_RES_PTR_SIZE)};
		}
		static RawTypeKind::Ptr newRawTy()
		{
			auto& elems = *(new PtrRawType());
			return RawTypeKind::Ptr{&elems};
		}
		RawTypeKind::Ptr cloneRaw() const
		{
			RawTypeKind::Ptr res = newRawTy();
			res->elem = elem.clone();
			res->ptrType = ptrType;
			return res;
		}
	};
	export void ::slu::parse::DelStructRawType::operator()(
	    StructRawType * it) const noexcept
	{
		delete it;
	}
	export void ::slu::parse::DelUnionRawType::operator()(
	    UnionRawType * it) const noexcept
	{
		delete it;
	}
	export void ::slu::parse::DelVariantRawType::operator()(
	    VariantRawType * it) const noexcept
	{
		delete it;
	}
	export void ::slu::parse::DelRefRawType::operator()(
	    RefRawType * it) const noexcept
	{
		delete it;
	}
	export void ::slu::parse::DelPtrRawType::operator()(
	    PtrRawType * it) const noexcept
	{
		delete it;
	}
	export void ::slu::parse::DelSliceRawType::operator()(
	    SliceRawType * it) const noexcept
	{
		delete it;
	}
	extern "C++"
	{
	export void ::slu::parse::DelExpr::operator()(
	    parse::Expr * it) const noexcept
	{
		delete it;
	}
	}

	export constexpr std::optional<lang::MpItmId>
	ResolvedType::getStructName() const
	{
		if (!std::holds_alternative<RawTypeKind::Struct>(base))
			return std::nullopt;
		return std::get<RawTypeKind::Struct>(base)->name;
	}
	export RawType cloneRawType(const RawType& t)
	{
		return ezmatch(t)(
		    varcase(const auto&) { return RawType{var}; },
		    varcase(const RawTypeKind::Unresolved&)->RawType {
			    throw std::runtime_error("Cannot clone unresolved type");
		    },
		    varcase(const RawTypeKind::Struct&) {
			    return RawType{var->cloneRaw()};
		    },
		    varcase(
		        const RawTypeKind::Union&) { return RawType{var->cloneRaw()}; },
		    varcase(const RawTypeKind::Variant&) {
			    return RawType{var->cloneRaw()};
		    },
		    varcase(
		        const RawTypeKind::Ref&) { return RawType{var->cloneRaw()}; },
		    varcase(
		        const RawTypeKind::Ptr&) { return RawType{var->cloneRaw()}; },
		    varcase(
		        const RawTypeKind::Slice&) { return RawType{var->cloneRaw()}; }

		);
	}
} //namespace slu::parse