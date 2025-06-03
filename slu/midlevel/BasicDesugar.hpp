/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <thread>
#include <variant>
#include <slu/lang/BasicState.hpp>
#include <slu/parser/State.hpp>
#include <slu/visit/Visit.hpp>
#include <slu/midlevel/Operator.hpp>

namespace slu::mlvl
{
	using DesugarCfg = decltype(parse::sluCommon);
	struct DesugarVisitor : visit::EmptyVisitor<DesugarCfg>
	{
		//TODO: Implement the conversion logic here
		//TODO: basic desugaring:
		//TODO: operators
		//TODO: auto-drop?
		//TODO: for/while/repeat loops

		bool preExpr(parse::Expression<Cfg>& itm) {
			using MultiOp = parse::ExprType::MULTI_OPERATION<Cfg>;
			if (std::holds_alternative<MultiOp>(itm.data))
			{
				MultiOp& ops = std::get<MultiOp>(itm.data);
				_ASSERT(itm.unOps.empty());
				_ASSERT(itm.postUnOps.empty());
				auto order = multiOpOrder<false>(ops);
				
				parse::Expression<Cfg> newExpr{};
				newExpr.place = itm.place;
				//newExpr.data = ;
				for (auto& i : order)
				{

				}

				//desugar operators into trait func calls!
				return true;
			}
			return false;
		}; 
	};

	inline void basicDesugar(parse::ParsedFileV<true>& itm)
	{
		DesugarVisitor vi{};
		visit::visitFile(vi, itm);
	}
}