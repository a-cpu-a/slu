// ============================================================================
// MODULE TREE BUILDER
// ============================================================================

/**
 * Represents a node in the module tree (Module, Function, Impl, etc.)
 */
class Module {
    constructor(name, astNode) {
        this.type = 'Module';
        this.name = name;
        this.astNode = astNode; // Reference to the original AST node
        this.parent = null;

        // Lists of child items
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
    }
}

/**
 * Visitor that builds a tree structure of module items.
 * - Treats functions as modules (nodes that can contain constants via params).
 * - Treats const params as constants.
 * - Extracts methods from Impl/Trait bodies.
 */
class ModuleTreeVisitor extends Visitor {
    constructor(name, node) {
        super();
        // Create a virtual root node
        this.root = new Module(name, node);
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
        if (this.currentScope && this.currentScope[category]) {
            this.currentScope[category].push(item);
            item.parent = this.currentScope;
        }
    }

    // ========================================================================
    // HELPER: PROCESS FUNCTION SCOPES
    // ========================================================================

    /**
     * Common logic for FunctionDecl (top-level) and FnExpr (inside Impl/Trait).
     * Creates a new function node, pushes it to the stack, and processes its contents.
     */
    processFunction(name, node) {
        const fnNode = new Module(name, node);
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
                throw new Error(`Expected importable name, in use statement`);
            }
        } else // if (node.variant.type == 'StarUseVariant') 
        {
            this.addItem('useStars', node);
            return;
        }

        const item = { name: s, node: node };
        this.addItem('uses', item);
    }

    visitConstDecl(node) {
        let name = "anon";
        // Try to extract name from pattern (UncondDestrPat)
        // UncondVarDestrPat has a name property
        if (node.pattern && node.pattern.name && node.pattern.name.name) {
            name = node.pattern.name.name;
        }
        const item = { name: name, node: node };
        this.addItem('constants', item);
        this.visitChildren(node);
    }

    visitFunctionDecl(node) {
        this.processFunction(node.name.name, node);
    }

    visitModDecl(node) {
        const modNode = new Module(node.name.name, node);
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