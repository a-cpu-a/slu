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
#include <slu/parser/OpTraits.hpp>
#include <slu/visit/Visit.hpp>
#include <slu/midlevel/Operator.hpp>

namespace slu::mlvl
{
	template<typename T>
	struct LazyCompute
	{
		std::optional<T> value;

		T& get(auto&& create)
		{
			if (!value.has_value())
				value = create();
			return *value;
		}
	};
	using DesugarCfg = decltype(parse::sluCommon);
	struct DesugarVisitor : visit::EmptyVisitor<DesugarCfg>
	{
		parse::BasicMpDb mpDb;
		LazyCompute<parse::MpItmId<Cfg>> unOpFuncs[(size_t)parse::UnOpType::ENUM_SIZE];
		LazyCompute<parse::MpItmId<Cfg>> postUnOpFuncs[(size_t)parse::PostUnOpType::ENUM_SIZE];
		LazyCompute<parse::MpItmId<Cfg>> binOpFuncs[(size_t)parse::BinOpType::ENUM_SIZE];

		// Note: Implicit bottom item: 'Any' or 'Slu'
		std::vector<std::string> abiStack;
		std::vector<bool> abiSafetyStack;
		std::vector<parse::StatList<Cfg>*> statListStack;

		//TODO: Implement the conversion logic here
		//TODO: basic desugaring:
		//TODO: [50%] operators
		//TODO: [_0%] auto-drop?
		//TODO: [_0%] for/while/repeat loops

		bool preFunctionInfo(parse::FunctionInfo<Cfg>& itm) 
		{
			if(itm.abi.empty())
				itm.abi = abiStack.empty() ? "Any" : abiStack.back();
			return false;
		}
		bool preStatList(parse::StatList<Cfg>& itm) 
		{
			statListStack.push_back(&itm);
			return false;
		}
		void postStatList(parse::StatList<Cfg>& itm) {
			statListStack.pop_back();
		}
		bool preExternBlock(parse::StatementType::ExternBlock<Cfg>& itm) 
		{
			abiStack.push_back(std::move(itm.abi));
			abiSafetyStack.push_back(itm.safety==parse::OptSafety::SAFE);

			visit::visitStatList(*this,itm.stats);

			abiStack.pop_back();
			abiSafetyStack.pop_back();
			return true;//Already visited the statements inside it
		}
		void postStat(parse::Statement<Cfg>& itm) 
		{
			if (std::holds_alternative<parse::StatementType::ExternBlock<Cfg>>(itm.data))
			{
				// Unwrap the extern block
				auto& block = std::get<parse::StatementType::ExternBlock<Cfg>>(itm.data);
				if (block.stats.empty())
				{
					itm.data = parse::StatementType::SEMICOLON{};
					return;
				}

				parse::StatList<Cfg> stats = std::move(block.stats);
				itm.place = stats.front().place;
				auto statData = std::move(stats.front().data);
				itm.data = std::move(statData);
				if (stats.size() == 1)
					return;
				// Insert the rest of the statements
				auto& statList = *statListStack.back();
				statList.insert(statList.end(), std::make_move_iterator(std::next(stats.begin()+1)), std::make_move_iterator(stats.end()));
			}
		}

		bool preExpr(parse::Expression<Cfg>& itm) {
			using MultiOp = parse::ExprType::MULTI_OPERATION<Cfg>;
			if (std::holds_alternative<MultiOp>(itm.data))
			{
				MultiOp& ops = std::get<MultiOp>(itm.data);
				_ASSERT(itm.unOps.empty());
				_ASSERT(itm.postUnOps.empty());
				auto order = multiOpOrder<false>(ops);
				
				std::vector<parse::Expression<Cfg>> expStack;
				//newExpr.data = ;
				for (auto& i : order)
				{
					switch (i.kind) {
					case OpKind::Expr:
					{
						parse::Expression<Cfg>& parent = i.index == 0 ? *ops.first : ops.extra[i.index - 1].second;

						parse::Expression<Cfg>& newExpr = expStack.emplace_back();
						newExpr.data = std::move(parent.data);
						newExpr.place = parent.place;
						break;
					}
					case OpKind::BinOp:
					{
						//create a new expression for the binary operator

						//TODO: special handling for 'and', 'or', '~~' (maybe turn it special at the ast level?).

						auto expr2 = std::move(expStack.back());
						expStack.pop_back();
						auto& expr1 = expStack.back();
						parse::Position place = expr1.place;

						parse::ExprType::FUNC_CALL<Cfg> call;
						parse::ArgsType::EXPLIST<Cfg> list;
						list.v.emplace_back(std::move(expr1));
						list.v.emplace_back(std::move(expr2));
						call.argChain.emplace_back(parse::MpItmId<Cfg>::newEmpty(), std::move(list));

						parse::BinOpType op = ops.extra[i.index-1].first;
						const size_t traitIdx = (size_t)op - 1; //-1 for none

						*call.val = parse::LimPrefixExprType::VAR<Cfg>{ .v = parse::Var<Cfg>{.base = parse::BaseVarType::NAME<Cfg>{
							.v = binOpFuncs[traitIdx].get([&] {
									lang::ModPath name;
									name.reserve(4);
									name.emplace_back("std");
									bool isOrd = op == parse::BinOpType::LESS_THAN
										|| op == parse::BinOpType::LESS_EQUAL
										|| op == parse::BinOpType::GREATER_THAN
										|| op == parse::BinOpType::GREATER_EQUAL;
									bool isEq = op == parse::BinOpType::EQUAL
										|| op == parse::BinOpType::NOT_EQUAL;
									if (isOrd || isEq)
									{
										name.emplace_back("cmp");
										if(isOrd)
											name.emplace_back("PartialOrd");
										else
											name.emplace_back("PartialEq");
									}
									else
									{
										name.emplace_back("ops");
										if(op==parse::BinOpType::RANGE_BETWEEN)
											name.emplace_back("Boundable");//TODO: choose the name!
										else
										name.emplace_back(parse::binOpTraitNames[traitIdx]);
									}
									name.emplace_back(parse::binOpNames[traitIdx]);
									return mpDb.getItm(name);
								})
						}} };


						//Turn the first (moved)expr in a function call expression
						expr1.data = std::move(call);
						expr1.place = place;
						break;
					}
					case OpKind::UnOp:
					case OpKind::PostUnOp:
					{
						//TODO: special handling for '.*', '?'. -> defer to after type checking / inference

						auto& opSrcExpr = expStack[i.index];

						auto& expr = expStack.back();
						parse::Position place = expr.place;
						//Wrap
						parse::MpItmId<Cfg> name;
						parse::Lifetime* lifetime = nullptr;
						{
							if(i.kind == OpKind::UnOp) {
								parse::UnOpItem& op = opSrcExpr.unOps[i.opIdx];
								const size_t traitIdx = (size_t)op.type - 1; //-1 for none

								name = unOpFuncs[traitIdx].get([&] {
									lang::ModPath name;
									name.reserve(4);
									name.emplace_back("std");
									name.emplace_back("ops");
									if(op.type==parse::UnOpType::RANGE_BEFORE)
										name.emplace_back("Boundable");//TODO: choose the name!
									else
									name.emplace_back(parse::unOpTraitNames[traitIdx]);
									name.emplace_back(parse::unOpNames[traitIdx]);
									return mpDb.getItm(name);
								});
								if (!op.life.empty())
								{
									_ASSERT(op.type == parse::UnOpType::TO_REF
										|| op.type == parse::UnOpType::TO_REF_MUT
										|| op.type == parse::UnOpType::TO_REF_CONST
										|| op.type == parse::UnOpType::TO_REF_SHARE);
									lifetime = &op.life;
								}
								//Else: its inferred, or doesnt exist
							}
							else if (i.kind == OpKind::PostUnOp) {
								parse::PostUnOpType op = opSrcExpr.postUnOps.at(i.opIdx);
								const size_t traitIdx = (size_t)op - 1; //-1 for none

								name = postUnOpFuncs[traitIdx].get([&] {
									//TODO: implement post-unop func name selection
									if (op == parse::PostUnOpType::RANGE_AFTER)
									{
										lang::ModPath name;
										name.reserve(4);
										name.emplace_back("std");
										name.emplace_back("ops");
										name.emplace_back("Boundable");//TODO: choose the name!
										name.emplace_back(parse::postUnOpNames[traitIdx]);
										return mpDb.getItm(name);
									}
									//TODO!!!
									return parse::MpItmId<Cfg>::newEmpty();
								});
							}
						}
						parse::ExprType::FUNC_CALL<Cfg> call;
						parse::ArgsType::EXPLIST<Cfg> list;
						list.v.emplace_back(std::move(expr));
						if (lifetime != nullptr)
						{
							parse::Expression<Cfg> lifetimeExpr;
							lifetimeExpr.place = place;
							lifetimeExpr.data = parse::ExprType::LIFETIME{ std::move(*lifetime) };
							list.v.emplace_back(std::move(lifetimeExpr));
						}
						call.argChain.emplace_back(parse::MpItmId<Cfg>::newEmpty(), std::move(list));
						*call.val = parse::LimPrefixExprType::VAR<Cfg>{ .v = parse::Var<Cfg>{.base = parse::BaseVarType::NAME<Cfg>{
							.v = name
						}} };

						//Turn the (moved)expr in a function call expression
						expr.data = std::move(call);
						expr.place = place;
						break;
					}
					};
				}

				//desugar operators into trait func calls!
				return true;
			}
			return false;
		}; 
	};

	inline void basicDesugar(parse::BasicMpDbData& mpDbData,parse::ParsedFileV<true>& itm)
	{
		DesugarVisitor vi{ {},parse::BasicMpDb{ &mpDbData } };

		visit::visitFile(vi, itm);
	}
}