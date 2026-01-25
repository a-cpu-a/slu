/*
    Slu language compiler, a computer program compiler.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
*/
module;
#include <memory>
#include <optional>
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
	using Sel = std::conditional_t<flag, TrueT, FalseT>;

	export template<bool boxed, class T> struct MayBox
	{
		Sel<boxed, T, std::unique_ptr<T>> v;

		T& get()
		{
			if constexpr (boxed)
				return *v;
			else
				return v;
		}
		const T& get() const
		{
			if constexpr (boxed)
				return *v;
			else
				return v;
		}

		T& operator*()
		{
			return get();
		}
		const T& operator*() const
		{
			return get();
		}

		T* operator->()
		{
			return &get();
		}
		const T* operator->() const
		{
			return &get();
		}
	};
	export template<bool boxed, class T> constexpr auto mayBoxFrom(T&& v)
	{
		if constexpr (boxed)
			return MayBox<true, T>(std::make_unique<T>(std::move(v)));
		else
			return MayBox<false, T>(std::move(v));
	}
	export template<class T> constexpr MayBox<false, T> wontBox(T&& v)
	{
		return MayBox<false, T>(std::move(v));
	}

	//Forward declare
	extern "C++"
	{
	export struct Stat;
	export struct Expr;
	}
	export using BoxExpr = std::unique_ptr<Expr>;
	export using ExprList = std::vector<Expr>;

	namespace FieldType
	{
		//For lua only! (currently)
		extern "C++"
		{
		export struct Expr2Expr;
		export struct Name2Expr;
		}
		export using parse::Expr;
	} //namespace FieldType
	namespace ExprType
	{
		export struct OpenRange
		{};
		export struct String
		{
			std::string v;
			ast::Position end;
		};

		// "Numeral"
		export using F64 = double;
		export using I64 = int64_t;

		//u64,i128,u128, for slu only
		export using U64 = uint64_t;
		export using P128 = Integer128<false>;
		export using M128 = Integer128<false, true>;
	} //namespace ExprType
	export using SubModPath = std::vector<std::string>;
} //namespace slu::parse