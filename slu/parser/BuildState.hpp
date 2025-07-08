/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <slu/parser/State.hpp>

namespace slu::parse
{
	template<AnyCfgable Cfg>
	inline std::unique_ptr<::slu::parse::LimPrefixExpr<Cfg>> mkLpe(::slu::parse::LimPrefixExpr<Cfg>&& v) {
		return std::make_unique<::slu::parse::LimPrefixExpr<Cfg>>(v);
	}
	template<AnyCfgable Cfg>
	inline std::unique_ptr<::slu::parse::LimPrefixExpr<Cfg>> mkLpeVar(::slu::parse::MpItmId<Cfg> name) {
		return ::slu::parse::mkLpe(
			::slu::parse::LimPrefixExprType::VAR<Cfg>{
			.v = ::slu::parse::Var<Cfg>{
				.base = ::slu::parse::BaseVarType::NAME<Cfg>{
					.v = name
		} } });
	}
}