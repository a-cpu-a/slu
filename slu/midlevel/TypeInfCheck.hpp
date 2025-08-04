/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <algorithm>
#include <slu/ext/CppMatch.hpp>
#include <slu/lang/BasicState.hpp>
#include <slu/parser/State.hpp>
#include <slu/parser/BuildState.hpp>
#include <slu/parser/OpTraits.hpp>
#include <slu/visit/Visit.hpp>
#include <slu/midlevel/ResolveType.hpp>

namespace slu::mlvl
{
	using TypeInfCheckCfg = decltype(parse::sluCommon);

	template <class T>
	inline bool sameCheck(const T& itm, const parse::ResolvedType& useTy,const auto& onSame)
	{
		return ezmatch(useTy.base)(
		varcase(const auto&) { return false; },
		varcase(const parse::RawTypeKind::TypeError) {
			return true;//poisioned, so pass forward.
		},
		varcase(const parse::RawTypeKind::Variant&) {
			for (const auto& i : var->options)
			{
				if (sameCheck(itm, i, onSame))
					return true;//Yep atleast one item is a valid thing.
			}
			return false;
		},
		varcase(const T&) {
			return onSame(itm, var);
		}
		);
	}
	inline bool rangeRangeSubtypeCheck(const parse::AnyRawRange auto itm, const parse::AnyRawRange auto useTy) {
		return itm.max <= useTy.max && itm.min >= useTy.min;
	}
	inline bool _compilerHackGt(auto a, auto b) {
		return a > b;
	}
	template <parse::AnyRawIntOrRange T>
	inline bool intRangeSubtypeCheck(const T itm, const parse::ResolvedType& useTy)
	{
		return ezmatch(useTy.base)(
		varcase(const auto&) { return false; },
		varcase(const parse::RawTypeKind::TypeError) {
			return true;//poisioned, so pass forward.
		},
		varcase(const parse::RawTypeKind::Variant&) {
			for (const auto& i : var->options)
			{
				if (intRangeSubtypeCheck(itm, i))
					return true;//Yep atleast one item is a valid thing.
			}
			return false;
		},
		varcase(const parse::AnyRawIntOrRange auto&) {
			using VarT = std::remove_cvref_t<decltype(var)>;
			constexpr bool itmIsInt = parse::AnyRawInt<T>;
			constexpr bool isInt = parse::AnyRawInt<VarT>;

			if constexpr (itmIsInt && isInt)
			{// Check for sign mismatch
				if constexpr (std::same_as<T, parse::RawTypeKind::Int64> && std::same_as<VarT, parse::RawTypeKind::Uint64>)
				{
					if (var > (uint64_t)INT64_MAX)
						return false;
					return itm == (int64_t)var;
				}
				else if constexpr (std::same_as<T, parse::RawTypeKind::Uint64> && std::same_as<VarT, parse::RawTypeKind::Int64>)
				{
					if (_compilerHackGt(itm, (uint64_t)INT64_MAX))
						return false;
					return (int64_t)itm == var;
				}
				else
					return itm == var;
			}
			else if constexpr (itmIsInt)
				return var.isInside(itm);
			else if constexpr (isInt)
				return itm.isOnly(var);
			else
				return rangeRangeSubtypeCheck(itm,var);
		}
		);
	}
	bool subtypeCheck(parse::BasicMpDb mpDb, const parse::ResolvedType& subty, const parse::ResolvedType& useTy);

	inline bool nameMatchCheck(parse::BasicMpDb mpDb,parse::MpItmIdV<true> subName, parse::MpItmIdV<true> useName)
	{
		if(subName == useName) return true;//Same name, so match.
		if (subName.empty()) return true;
		if (useName.empty()) return false;//Named -/> unnamed.
		//TODO: upcasts.
		return false;
	}
	template <class T>
	inline bool nearExactCheckDeref(const T& subty, const parse::ResolvedType& useTy)
	{
		if (!std::holds_alternative<T>(useTy.base)) return false;
		return subty->nearlyExact(*std::get<T>(useTy.base));
	}
	inline bool subtypeCheckRefChain(const parse::RawTypeKind::RefChain& var, const parse::ResolvedType& useTy)
	{
		using T = parse::RawTypeKind::RefChain;
		return sameCheck<T>(var, useTy, [&](const T& var, const T& useTy) {
			if (var->chain != useTy->chain)
				return false;
			return nearExactCheck(var->elem, useTy->elem);
		});
	}
	inline bool subtypeCheckRefSlice(const parse::RawTypeKind::RefSlice& var, const parse::ResolvedType& useTy)
	{
		using T = parse::RawTypeKind::RefSlice;
		return sameCheck<T>(var, useTy, [&](const T& var, const T& useTy) {
			if (var->refType != useTy->refType)
				return false;
			if (var->elem.outerSliceDims != useTy->elem.outerSliceDims)
				return false;
			return nearExactCheck(var->elem, useTy->elem);
		});
	}

	//ignores outerSliceDims of either side, as if checking the slices element type.
	inline bool nearExactCheck(const parse::ResolvedType& subty, const parse::ResolvedType& useTy)
	{
		if(std::holds_alternative<parse::RawTypeKind::TypeError>(useTy.base))
			return true;//poisioned, so pass forward.

		//This should already be true, if all parts of the near exact check are correct.
		//if (subty.size != useTy.size)
		//	return false;

		return ezmatch(subty.base)(
		[&]<class T>(const T& var) {
			if(!std::holds_alternative<T>(useTy.base)) return false;
			return var == std::get<T>(useTy.base);
		},
		varcase(const parse::RawTypeKind::TypeError) {
			return true;
		},
		varcase(const parse::RawTypeKind::Inferred) ->bool{
			throw std::runtime_error("TODO: error logging, Found Inferred type in near exact type check");
		},
		varcase(const parse::RawTypeKind::Unresolved&)->bool {
			throw std::runtime_error("TODO: error logging, Found unresolved type in near exact type check");
		},

		varcase(const parse::RawTypeKind::Variant&) {
			return nearExactCheckDeref(var,useTy);
		},
		varcase(const parse::RawTypeKind::Struct&) {
			return nearExactCheckDeref(var,useTy);
		},
		varcase(const parse::RawTypeKind::Union&) {
			return nearExactCheckDeref(var,useTy);
		},

		varcase(const parse::RawTypeKind::RefChain&) {
			return subtypeCheckRefChain(var, useTy);
		},
		varcase(const parse::RawTypeKind::RefSlice&) {
			return subtypeCheckRefSlice(var, useTy);
		}
		);
	}
	inline bool subtypeCheck(parse::BasicMpDb mpDb,const parse::ResolvedType& subty, const parse::ResolvedType& useTy)
	{
		if (subty.outerSliceDims != useTy.outerSliceDims)
			return false;//TODO: allow variant here.
		if (subty.outerSliceDims != 0)//both have the same non-0 slice size.
			return nearExactCheck(subty, useTy);//TODO: allow variant here.

		return ezmatch(subty.base)(
		varcase(const parse::RawTypeKind::Unresolved&)->bool {
			throw std::runtime_error("Found unresolved type in subtype check");
		},
		varcase(const parse::RawTypeKind::Inferred)->bool {
			throw std::runtime_error("Found inferred type in subtype check");
		},
		varcase(const parse::RawTypeKind::TypeError) {
			return true;//poisioned, so pass forward.
		},
		varcase(const parse::RawTypeKind::String&) {
			using T = parse::RawTypeKind::String;
			return sameCheck<T>(var, useTy, std::equal_to<T>{});//TODO: subtyping into &str.
		},
		varcase(const parse::RawTypeKind::Float64) {
			using T = parse::RawTypeKind::Float64;
			return sameCheck<T>(var, useTy, std::equal_to<T>{});//TODO: allow f64 as the type too (needs to be impl first).
		},

		varcase(const parse::RawTypeKind::Variant&) {
			for (const auto& i : var->options)
			{
				if (!subtypeCheck(mpDb, i, useTy))
					return false;
			}
			return true;
		},
		varcase(const parse::RawTypeKind::Union&) {
			using T = parse::RawTypeKind::Union;
			return sameCheck<T>(var, useTy, [&](const T& var, const T& useTy) {
				if(var->fields.size() != var->fields.size())
					return false;
				if (!nameMatchCheck(mpDb, var->name, useTy->name))
					return false;

				for (size_t i = 0; i < var->fields.size(); i++)
				{
					if(var->fields[i].outerSliceDims != useTy->fields[i].outerSliceDims)
						return false;
					if(!nearExactCheck(var->fields[i], useTy->fields[i]))
						return false;
					if(var->fieldNames[i] != useTy->fieldNames[i])
						return false;
				}
				return true;
			});
		},
		varcase(const parse::RawTypeKind::Struct&) {
			using T = parse::RawTypeKind::Struct;
			return sameCheck<T>(var, useTy, [&](const T& var, const T& useTy) {
				if (!nameMatchCheck(mpDb, var->name, useTy->name))
					return false;
				for (size_t i = 0; i < var->fields.size(); i++)
				{
					const parse::ResolvedType& ty = var->fields[i];
					const std::string& name = var->fieldNames[i];
					//TODO: locate same field in other type & subtype check it.
				}
			});
		},

		varcase(const parse::RawTypeKind::RefChain&) {
			return subtypeCheckRefChain(var, useTy);
		},
		varcase(const parse::RawTypeKind::RefSlice&) {
			return subtypeCheckRefSlice(var, useTy);
		},

		varcase(const parse::AnyRawIntOrRange auto&) {
			return intRangeSubtypeCheck(var, useTy);
		}
		);
	}


	//TODO: fix string expressions being fully stolen.
	//TODO: create TmpVar's for func call results, table indexing.
	using TmpVar = uint64_t;
	using VisitTypeBuilder = std::variant<parse::ResolvedType,const parse::ResolvedType*, parse::LocalId, TmpVar>;

	struct SubTySide
	{
		std::vector<TmpVar> tmpLocals;
		std::vector<parse::LocalId> locals;
		std::vector<parse::ResolvedType> tys;
		std::vector<const parse::ResolvedType*> tyRefs;

		void clear()
		{
			tmpLocals.clear();
			locals.clear();
			tys.clear();
			tyRefs.clear();
		}
	};
	struct LocalVarInfo
	{
		std::vector<lang::LocalObjId> fields;
		//traits?
		//???

		SubTySide use;//Requirements when used.
		SubTySide edit;//Requirements when writen to.

		bool boolLike : 1 = false;//part of use.
		bool taken    : 1 = false;
		bool resolved : 1 = false;

		constexpr parse::ResolvedType* resolvedType()
		{
			if (!resolved)return nullptr;
			return &edit.tys[0];
		}
		parse::ResolvedType& resolveNoCheck(parse::ResolvedType&& t)
		{
			_ASSERT(!resolved);
			//use.clear();
			fields.clear();
			edit.clear();
			resolved = true;
			return edit.tys.emplace_back(std::move(t));
		}
		void requireBoolLike(parse::BasicMpDb mpDb)
		{
			if(resolved && !boolLike && !resolvedType()->isBool(mpDb))
				throw std::runtime_error("TODO: error logging, found non bool expr");

			boolLike = true;
		}
	};
	using LocalVarList = std::vector<LocalVarInfo>;

	struct TypeInfCheckVisitor : visit::EmptyVisitor<TypeInfCheckCfg>
	{
		using Cfg = TypeInfCheckCfg;
		static constexpr bool isSlu = Cfg::settings() & ::slu::parse::sluSyn;

		parse::BasicMpDb mpDb;
		std::vector<parse::Locals<Cfg>*> localsStack;
		std::vector<LocalVarList> localsDataStack;
		std::vector<LocalVarList> tmpLocalsDataStack;

		std::vector<VisitTypeBuilder> exprTypeStack;

		LocalVarInfo& localVar(const parse::LocalId id) {
			return localsDataStack.back()[id.v];
		}
		LocalVarInfo& localVar(const TmpVar id) {
			return tmpLocalsDataStack.back()[id];
		}

		void requireAsBool(const VisitTypeBuilder& t)
		{
			ezmatch(t)(
			varcase(const parse::ResolvedType&) {
				if (!var.isBool(mpDb))
					throw std::runtime_error("TODO: error logging, found non bool expr");
			},
			varcase(const parse::ResolvedType*) {
				if (!var->isBool(mpDb))
					throw std::runtime_error("TODO: error logging, found non bool expr");
			},
			varcase(const parse::LocalId) {
				localVar(var).requireBoolLike(mpDb);
			},
			varcase(const TmpVar) {
				localVar(var).requireBoolLike(mpDb);
			}
			);
		}
		void requireUseTy(const VisitTypeBuilder& t,const parse::ResolvedType& ty)
		{
			ezmatch(t)(
			varcase(const parse::ResolvedType&) {
				if(!subtypeCheck(mpDb,var, ty))
					throw std::runtime_error("TODO: error logging, found non matching type expr");
			},
			varcase(const parse::ResolvedType*) {
				if (!subtypeCheck(mpDb, *var, ty))
					throw std::runtime_error("TODO: error logging, found non matching type expr");
			},
			varcase(const parse::LocalId) {
				localVar(var).use.tyRefs.push_back(&ty);
			},
			varcase(const TmpVar) {
				localVar(var).use.tyRefs.push_back(&ty);
			}
			);
		}

		bool preExpr(parse::Expr<Cfg>& itm) 
		{
			exprTypeStack.emplace_back();
			return false;
		}

		template<class RawT>
		bool handleConstType(auto&& v)
		{
			exprTypeStack.emplace_back(parse::ResolvedType::getConstType(RawT{ std::move(v) }));
			return false;
		}
		void editLocalVar(parse::LocalId itm)
		{
			VisitTypeBuilder& editTy = exprTypeStack.back();
			ezmatch(editTy)(
			varcase(const auto&) {},
			varcase(const parse::LocalId) {
				localVar(var).use.locals.push_back(itm);
			},
			varcase(const TmpVar) {
				localVar(var).use.locals.push_back(itm);
			}
			);
			ezmatch(editTy)(
			varcase(parse::ResolvedType&) {
				localVar(itm).edit.tys.emplace_back(std::move(var));
			},
			varcase(const parse::ResolvedType*) {
				localVar(itm).edit.tyRefs.emplace_back(var);
			},
			varcase(const parse::LocalId) {
				localVar(itm).edit.locals.emplace_back(var);
			},
			varcase(const TmpVar) {
				localVar(itm).edit.tmpLocals.emplace_back(var);
			}
			);
			exprTypeStack.pop_back();
		}

		bool preF64(parse::ExprType::F64 itm) {
			return handleConstType<parse::RawTypeKind::Float64>(itm);
		}

		bool preOpenRange(parse::ExprType::OpenRange itm) {
			throw std::runtime_error("TODO: type-check/inferr open range expressions.");
		}
		bool preI64(parse::ExprType::I64 itm) {
			return handleConstType<parse::RawTypeKind::Int64>(itm);
		}
		bool preU64(parse::ExprType::U64 itm) {
			return handleConstType<parse::RawTypeKind::Uint64>(itm);
		}
		bool preM128(parse::ExprType::M128 itm) {
			return handleConstType<parse::RawTypeKind::Neg128>(itm);
		}
		bool preP128(parse::ExprType::P128 itm) {
			return handleConstType<parse::RawTypeKind::Pos128>(itm);
		}
		bool preExprString(parse::ExprType::String& itm) {
			return handleConstType<parse::RawTypeKind::String>(std::move(itm.v));//Steal it as converter will use the type anyway.
		}

		//Restrictions.
		void postAnyCond(parse::Expr<Cfg>& itm) {
			requireAsBool(exprTypeStack.back());
			exprTypeStack.pop_back();
		}
		void postCanonicLocal(parse::StatementType::CanonicLocal& itm) {
			editLocalVar(itm.name);//TODO: restrict the type to exactly that? (unless it is inferr)
		}
		void postFuncCallStat(parse::StatementType::FuncCall<Cfg>& itm) {
			if(itm.argChain.size() != 1)
				throw std::runtime_error("TODO: type inference for complex func call args.");
			if(!std::holds_alternative<parse::ArgsType::ExprList<Cfg>>(itm.argChain[0].args))
				throw std::runtime_error("TODO: type inference for func call with complex args.");

			if(!std::holds_alternative<parse::LimPrefixExprType::VAR<Cfg>>(*itm.val))
				throw std::runtime_error("TODO: type inference for func call on expr.");
			parse::Var<Cfg>& funcVar = std::get<parse::LimPrefixExprType::VAR<Cfg>>(*itm.val).v;
			if (!funcVar.sub.empty())
				throw std::runtime_error("TODO: type inference for sub variables in func-call statement.");
			if (!std::holds_alternative<parse::BaseVarType::NAME<Cfg>>(funcVar.base))
				throw std::runtime_error("TODO: type inference for func call on non-global var.");

			parse::MpItmId<Cfg> funcName = std::get<parse::BaseVarType::NAME<Cfg>>(funcVar.base).v;
			const parse::ItmType::Fn& funcItm = std::get<parse::ItmType::Fn>(mpDb.data->getItm(funcName));

			parse::ArgsType::ExprList<Cfg>& args = std::get<parse::ArgsType::ExprList<Cfg>>(itm.argChain[0].args);
			//Restrict arg exprs to match types in funcItm.
			for (size_t i = args.size(); i > 0; i++)
			{
				const parse::ResolvedType& ty = funcItm.args[i];
				requireUseTy(exprTypeStack.back(), ty);
				exprTypeStack.pop_back();
			}
			//Make temp var for func result, also add editType for it.
			//const TmpVar tmpVar = TmpVar(tmpLocalsDataStack.back().size());
			//tmpLocalsDataStack.back().emplace_back().editTys.emplace_back(&funcItm.ret);
			//
			//exprTypeStack.emplace_back(tmpVar);
		}
		void postAssign(parse::StatementType::Assign<Cfg>& itm)
		{
			size_t count = itm.vars.size();
			for (size_t i = count; i > 0; i--)
			{
				parse::Var<Cfg>& var = itm.vars[i-1];
				if(!var.sub.empty())
					throw std::runtime_error("TODO: type inference for sub variables in assign statement.");

				ezmatch(var.base)(
				varcase(parse::BaseVarType::NAMEv<true>&) {
					throw std::runtime_error("TODO: type check global assign statement.");
				},
				varcase(parse::BaseVarType::ExprV<true>&) {
					throw std::runtime_error("TODO: type inference for expr-var in assign statement.");
				},
				varcase(parse::BaseVarType::Local&) {
					editLocalVar(var);
				},

				varcase(parse::BaseVarType::Root&) {
					throw std::runtime_error("TODO better logging: cant assign to mp root (:>).");
				}
				);
			}
		}

		//Allow any type.
		void postDrop(parse::StatementType::Drop<Cfg>&) {
			exprTypeStack.pop_back();
		}

		//Ignored.
		bool preCanonicGlobal(parse::StatementType::CanonicGlobal&) {
			return true;
		}
		//Stack stuff.
		bool preLocals(parse::Locals<Cfg>& itm)
		{
			localsStack.push_back(&itm);
			localsDataStack.emplace_back();
			tmpLocalsDataStack.emplace_back();
			return false;
		}

		void visitSubTySide(SubTySide& side, auto&& visitor)
		{
			for (auto& i : side.tmpLocals)
				visitor(resolveLocal(localVar(i)));
			for (auto& i : side.locals)
				visitor(resolveLocal(localVar(i)));
			for (auto& i : side.tys)
				visitor(i);
			for (auto& i : side.tyRefs)
				visitor(*i);
		}

		// scales: O(N^3) // n(1+2n^2+3n)/6 // n is number of assignments*avg1OrVariantSize.
		void addSubTypeToList(std::vector<const parse::ResolvedType*>& types, const parse::ResolvedType& editTy)
		{
			for (const parse::ResolvedType* j : types)
			{
				if (subtypeCheck(mpDb, editTy, *j))
					return;//Already found a supertype of it.
			}
			types.emplace_back(&editTy);
			//TODO: maybe look for subtypes of editTy, or atleast unify them.
		}
		void visitTypeForInference(bool& poison,
			std::vector<const parse::ResolvedType*>& types,
			std::vector<parse::Range129>& intRanges,
			const parse::ResolvedType& editTy) 
		{
			if (!editTy.isComplete())
				throw std::runtime_error("TODO: error logging, found incomplete type expr");
			if (editTy.outerSliceDims != 0)
			{
				addSubTypeToList(types, editTy);
				return;
			}
			ezmatch(editTy.base)(
				varcase(const parse::RawTypeKind::Unresolved&) {
				throw std::runtime_error("TODO: error logging, found unresolved type expr");
			},
				varcase(const parse::RawTypeKind::Inferred) {
				throw std::runtime_error("TODO: error logging, found inferred type expr");
			},
				varcase(const parse::RawTypeKind::TypeError) {
				poison = true;
			},
				varcase(const parse::RawTypeKind::String&) {
				addSubTypeToList(types, editTy);
			},
				varcase(const parse::RawTypeKind::Union&) {
				addSubTypeToList(types, editTy);
			},
				varcase(const parse::RawTypeKind::Struct&) {
				addSubTypeToList(types, editTy);//Todo: potential unification.
			},
				varcase(const parse::RawTypeKind::Variant&) {
				for (auto& j : var->options)
					visitTypeForInference(poison, types, intRanges, j);
			},
				varcase(const parse::RawTypeKind::RefChain&) {
				addSubTypeToList(types, editTy);
			},
				varcase(const parse::RawTypeKind::RefSlice&) {
				addSubTypeToList(types, editTy);
			},

				varcase(const parse::RawTypeKind::Float64) {
				//TODO: float+float -> float-type... or float+float -> float || float?
			},
				[&]<parse::AnyRawIntOrRange T>(const T & var) {
				if constexpr (parse::AnyRawInt<T>)
					intRanges.emplace_back(parse::range129FromInt(var));
				else if constexpr (std::same_as<T, parse::RawTypeKind::Range64>)
					intRanges.emplace_back(parse::range129From64(var));
				else
					intRanges.emplace_back(var);
			}
				);
		}

		void checkLocals(std::span<LocalVarInfo> locals)
		{
			for (LocalVarInfo& i : locals)
			{
				if (i.resolved)
					continue;
				if(i.taken)
					throw std::runtime_error("TODO: error logging, variable type depends on itself, cant inferr it");
				i.taken = true;
				// Resolve its type

				if (i.boolLike)
				{// "true", "false" or "bool"
					bool canTrue = false;
					bool canFalse = false;
					bool poison = false;

					//First resolve those things & also check types a bit.

					visitSubTySide(i.edit,
						[&](const parse::ResolvedType& otherTy) {
							if(std::holds_alternative<parse::RawTypeKind::TypeError>(otherTy.base))
							{
								poison = true;
								return;
							}

							if (!otherTy.isComplete())
								throw std::runtime_error("TODO: error logging, found incomplete type expr");
							if(otherTy.size>1)
								throw std::runtime_error("TODO: error logging, found non bool expr");
							auto tyNameOpt = otherTy.getStructName();
							if (!tyNameOpt)
								throw std::runtime_error("TODO: error logging, found non bool expr");
							parse::MpItmIdV<true> tyName = *tyNameOpt;
							if (tyName == mpDb.data->getItm({ "std","bool" }))
								canTrue = canFalse = true;
							else if (tyName == mpDb.data->getItm({ "std","bool", "true" }))
								canTrue = true;
							else if (tyName == mpDb.data->getItm({ "std","bool", "false" }))
								canFalse = true;
							else
								throw std::runtime_error("TODO: error logging, found non bool expr");
						}
					);

					if(!(canTrue || canFalse))
					{
						poison = true;
						throw std::runtime_error("TODO: error logging, found non bool type");
					}

					if (poison)
						i.resolveNoCheck(parse::ResolvedType::newError());
					else
						i.resolveNoCheck(parse::ResolvedType::getBool(mpDb,canTrue, canFalse));
				}
				else
				{//resolve it (find a type, that is a subtype of all the edit types).
					bool poison = false;
					std::vector<const parse::ResolvedType*> types;
					std::vector<parse::Range129> intRanges;
					visitSubTySide(i.edit,
						[&](const parse::ResolvedType& editTy) {
							visitTypeForInference(poison, types, intRanges, editTy);
						});

					if (poison)
						i.resolveNoCheck(parse::ResolvedType::newError());
					else
					{
						parse::Range129 fullIntRange;
						if(!intRanges.empty())
						{
							// Sort by start, end doesnt matter, cuz we only care about overlapping/adjacency.
							std::sort(intRanges.begin(), intRanges.end(),
								[&](const parse::Range129& a, const parse::Range129& b) {
									return parse::r129Get(a, [&](const auto& aRange) {
										return parse::r129Get(b, [&](const auto& bRange) {
											return aRange.min < bRange.min;
											});
									});
								}
							);
							bool first = true;
							for (const auto& j : intRanges)
							{
								if (first)
								{
									fullIntRange = j;
									continue;
								}
								parse::r129Get(j, [&](const auto& jRange) {
									parse::r129Get(fullIntRange, [&](const auto& fullRange) {
										if (jRange.min.lteOtherPlus1(fullRange.max))
										{// Overlapping or adjacent, so merge them.
											const bool maxBigger = fullRange.max < jRange.max;
											const bool minSmaller = fullRange.min > jRange.min;

											if (maxBigger && minSmaller)
												fullIntRange = j;
											else 
											{
												if constexpr (
													!(std::same_as<decltype(fullRange.min), parse::Integer128<false, false>>
														&& std::same_as<decltype(jRange.max), parse::Integer128<false, true>>)
													)
												{
													if (maxBigger)
													{
														fullIntRange = parse::r129From(fullRange.min, jRange.max);
														return;
													}
												}
												if constexpr (
													!(std::same_as<decltype(jRange.min), parse::Integer128<false, false>>
														&& std::same_as<decltype(fullRange.max), parse::Integer128<false, true>>)
													)
												{
													if (minSmaller)
														fullIntRange = parse::r129From(jRange.min, fullRange.max);
												}
											}
											return;
										}
										// No overlap.
										throw std::runtime_error("TODO: error logging, found non overlapping int ranges in type inference");
									});
								});
							}
						}
						//Make a new type that is a subtype of all the edit types (a variant or just the first element)
						_ASSERT(!types.empty());
						if (types.size() == 1)
							i.resolveNoCheck(types[0]->clone());
						else
						{
							parse::RawTypeKind::Variant v = parse::VariantRawType::newRawTy();
							for (const parse::ResolvedType* j : types)
								v->options.emplace_back(j->clone());
							if (!intRanges.empty())
							{
								parse::r129Get(fullIntRange, [&](const auto& fullRange) {
									v->options.emplace_back(parse::ResolvedType::newIntRange(fullRange));
								});
							}
							size_t sz = v->calcSize();
							parse::ResolvedType res;
							res.base = std::move(v);
							res.size = sz;
							i.resolveNoCheck(std::move(res));
						}
					}
				}
				//Check use's

				visitSubTySide(i.use,
					[&](const parse::ResolvedType& useTy) {
						if (!useTy.isComplete())
							throw std::runtime_error("TODO: error logging, found incomplete type expr");
						// Check if the resolved type is a subtype of useTy.
						if (!subtypeCheck(mpDb, *i.resolvedType(), useTy))
							throw std::runtime_error("TODO: error logging, found type that is not a subtype of use type");
					}
				);
				i.use.clear();

				i.taken = false;
			}
		}
		parse::ResolvedType& resolveLocal(LocalVarInfo& local)
		{
			if (local.resolved)
				return *local.resolvedType();
			
			checkLocals(std::span<LocalVarInfo>{ &local,1 });
			return *local.resolvedType();
		}

		void postLocals(parse::Locals<Cfg>& itm) {
			//Check all restrictions.

			checkLocals(tmpLocalsDataStack.back());
			checkLocals(localsDataStack.back());
			//TODO: Export the types, so conv can use them.

			localsStack.pop_back();
			localsDataStack.pop_back();
			tmpLocalsDataStack.pop_back();
		}
	};

	inline void typeInferrAndCheck(parse::BasicMpDbData& mpDbData, lang::MpItmIdV<true> modName, parse::StatListV<true>& module)
	{
		TypeInfCheckVisitor vi{ {},parse::BasicMpDb{ &mpDbData } };

		for (auto& i : module)
			visit::visitStat(vi, i);
		_ASSERT(vi.exprTypeStack.empty());
	}
}