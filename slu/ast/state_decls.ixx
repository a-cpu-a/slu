module;
/*
** See Copyright Notice inside Include.hpp
*/
#include <optional>
#include <memory>
#include <span>
#include <string>
#include <vector>

export module slu.ast.state_decls;

import slu.big_int;
import slu.ast.pos;
import slu.lang.basic_state;
import slu.parse.input;

namespace slu::parse //TODO: ast
{
	export template<bool flag, class FalseT, class TrueT>
	using Sel = std::conditional_t<flag, TrueT,FalseT>;

	export template<bool boxed, class T>
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

		T* operator->() { return &get(); }
		const T* operator->() const { return &get(); }
	};
	export template<bool boxed, class T>
	constexpr auto mayBoxFrom(T&& v)
	{
		if constexpr (boxed)
			return MayBox<true, T>(std::make_unique<T>(std::move(v)));
		else
			return MayBox<false, T>(std::move(v));
	}
	export template<class T>
	constexpr MayBox<false, T> wontBox(T&& v) {
		return MayBox<false, T>(std::move(v));
	}

	//Forward declare
	extern "C++" {
		export struct Stat;
		export struct Expr;
	}
	export using BoxExpr = std::unique_ptr<Expr>;
	export using ExprList = std::vector<Expr>;

	namespace FieldType
	{
		//For lua only! (currently)
		extern "C++" {
			export struct Expr2Expr;
			export struct Name2Expr;
		}
		export using parse::Expr;
	}
	namespace ExprType
	{
		export struct OpenRange {};
		export struct String { std::string v; ast::Position end; };

		// "Numeral"
		export using F64 = double;
		export using I64 = int64_t;

		//u64,i128,u128, for slu only
		export using U64 = uint64_t;
		export using P128 = Integer128<false>;
		export using M128 = Integer128<false, true>;
	}
	export using SubModPath = std::vector<std::string>;
}