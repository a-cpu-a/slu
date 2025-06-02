/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>
#include <format>
#include <vector>

#include <slu/parser/State.hpp>
#include <slu/Settings.hpp>

namespace slu::visit
{
	template<parse::AnySettings _SettingsT = parse::Setting<void>>
	struct EmptyVisitor
	{
		using SettingsT = _SettingsT;
		using Cfg = parse::VecInput<SettingsT>;//TODO: swap with dummy settings cfg holder
		constexpr static SettingsT settings() { return SettingsT(); }

		constexpr EmptyVisitor(SettingsT) {}
		constexpr EmptyVisitor() = default;

		bool preName(parse::MpItmId<Cfg>& itm)
		{
			return false;
		}
		void postName(parse::MpItmId<Cfg>& itm)
		{}

		bool preString(std::span<char>& itm)
		{
			return false;
		}
		void postString(std::span<char> itm)
		{}

		bool preFile(parse::ParsedFile<Cfg>& itm)
		{
			return false;
		}
		void postFile(parse::ParsedFile<Cfg>& itm)
		{}

		bool preBlock(parse::Block<Cfg>& itm)
		{
			return false;
		}
		void postBlock(parse::Block<Cfg>& itm)
		{}

		bool preVar(parse::Var<Cfg>& itm)
		{
			return false;
		}
		void postVar(parse::Var<Cfg>& itm)
		{}

		bool preExpression(parse::Expression<Cfg>& itm)
		{
			return false;
		}
		void postExpression(parse::Expression<Cfg>& itm)
		{}

		bool preVarList(std::span<parse::Var<Cfg>> itm)
		{
			return false;
		}
		void sepVarList(std::span<parse::Var<Cfg>> list, parse::Var<Cfg>& itm)
		{}
		void postVarList(std::span<parse::Var<Cfg>> itm)
		{}

		bool preExpList(std::span<parse::Expression<Cfg>> itm)
		{
			return false;
		}
		void sepExpList(std::span<parse::Expression<Cfg>> list, parse::Expression<Cfg>& itm)
		{}
		void postExpList(std::span<parse::Expression<Cfg>> itm)
		{}
	};

	/*
		(bool"shouldStop" pre_) post_
		sep_ -> commas ::'s etc
	*/
	template<class T>
	concept AnyVisitor = parse::AnyCfgable<T> && std::is_base_of_v<EmptyVisitor<typename T::SettingsT>, T>;
}