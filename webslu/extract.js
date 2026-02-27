
class Local {
    constructor(name) {
        this.type = "Local"
        this.name = name;
    }
}

class Scope {
    constructor(type, astNode) {
        this.type = type;

        this.astNode = astNode;

        this.uses = [];
        this.useStars = [];
        this.constants = [];
        this.functions = [];
        this.traits = [];
        this.impls = [];
        this.structs = [];
        this.enums = [];
        this.unions = [];
        this.modules = [];

        this.locals = []; // Array of Local
        this.anonScopes = [];
        this.labeledScopes = [];
    }
}
class CrateScope extends Scope {
    constructor(owner, name) {
        super("CrateScope", null)
        this.owner = owner;
        this.name = name;
    }
}
class ModuleScope extends Scope {
    constructor(name, astNode) {
        super("ModuleScope", astNode)
        this.name = name
    }
}
class AnonScope extends Scope {
    constructor(id, astNode) {
        super("AnonScope", astNode)
        this.id = id
    }
}
class LabeledScope extends Scope {
    constructor(lbl, astNode) {
        super("LabeledScope", astNode)
        this.lbl = lbl
    }
}
class NamedScope extends Scope {
    constructor(name, astNode) {
        super("NamedScope", astNode)
        this.name = name
    }
}

class ModuleTreeVisitor extends Visitor {
    constructor(name, node) {
        super();
        // Create a virtual root node
        this.root = new ModuleScope(name, node);
        this.scopeStack = [this.root];
    }

    /** Returns the current Module being populated */
    get currentScope() {
        return this.scopeStack[this.scopeStack.length - 1];
    }

    /**
     * Helper to add a child item to the current scope.
     * Automatically sets the parent pointer.
     */
    addItem(category, item) {
        this.currentScope[category].push(item);
    }
    /**
     * Common logic for FunctionDecl (top-level) and FnExpr (inside Impl/Trait).
     * Creates a new function node, pushes it to the stack, and processes its contents.
     */
    processFunction(name, node) {
        const fnNode = new NamedScope(name, node);
        this.addItem('functions', fnNode);

        this.scopeStack.push(fnNode);
        // Visit params (to find const params) and body
        this.visitChildren(node);
        this.scopeStack.pop();
    }

    // ========================================================================
    // GLOBAL DECLARATIONS (GlobStat)
    // ========================================================================

    visitUseDecl(node) {
        let s;
        if (node.variant.type == 'UseAs') {
            s = node.variant.name.name;
        } else if (node.variant.type == 'SimpleUseVariant') {
            //last segment
            if (node.path.segments.length !== 0)
                s = node.path.segments[node.path.segments.length - 1].name.name;
            else if (node.path.root.type === 'ModPathRootName') {
                s = node.path.root.name.name;
            } else {
                throw new Error(`Expected importable name in use statement`);
            }
        } else // if (node.variant.type == 'StarUseVariant') 
        {
            this.addItem('useStars', node);
            return;
        }

        const item = { name: s, node: node };
        this.addItem('uses', item);
    }

    visitVarPat(p) {
    }

    visitConstDecl(node) {
        let name = "anon";

        this.visitVarPat(node.pattern)

        const item = { name: name, node: node };
        this.addItem('constants', item);
        this.visit(node.value)
    }

    visitFunctionDecl(node) {
        this.processFunction(node.name.name, node);
    }

    visitModDecl(node) {
        const modNode = new ModuleScope(node.name.name, node);
        this.addItem('modules', modNode);

        this.scopeStack.push(modNode);
        this.visitChildren(node); // Visits 'chunk' (GlobStat[])
        this.scopeStack.pop();
    }

    visitTraitDecl(node) {
        const traitNode = new Module(node.name.name, node);
        this.addItem('traits', traitNode);

        this.scopeStack.push(traitNode);
        this.visitChildren(node);
        this.scopeStack.pop();
    }

    visitImplDecl(node) {

        const implMod = new Module('', node)
        this.addItem('impls', implMod);

        this.scopeStack.push(implMod);
        this.visitChildren(node);
        this.scopeStack.pop();
    }

    visitStructDecl(node) {
        const structMod = new Module(node.name.name, node)
        this.addItem('structs', structMod);

        this.scopeStack.push(structMod);
        this.visitChildren(node); // Visits params (const params) and body
        this.scopeStack.pop();
    }

    visitEnumDecl(node) {
        const enumNode = new Module(node.name.name, node);
        this.addItem('enums', enumNode);

        this.scopeStack.push(enumNode);
        this.visitChildren(node);
        this.scopeStack.pop();
    }

    visitUnionDecl(node) {
        const unionNode = new Module(node.name.name, node);
        this.addItem('unions', unionNode);

        this.scopeStack.push(unionNode);
        this.visitChildren(node);
        this.scopeStack.pop();
    }

    // ========================================================================
    // PARAMS & CONSTANTS
    // ========================================================================

    visitTypedParam(node) {
        // Treat const params like constants
        if (node.constKw.present) {
            const item = { name: node.name.name, node: node };
            this.addItem('constants', item);
        }
        // Visit children (e.g. the type expression)
        this.visitChildren(node);
    }
}