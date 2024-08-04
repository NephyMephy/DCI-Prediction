import { createRootRoute, Link, Outlet } from "@tanstack/react-router";
import { TanStackRouterDevtools } from "@tanstack/router-devtools";

export const Route = createRootRoute({
  component: () => (
    <>
      <div className="p-2 flex gap-2 border-b border-b-blue-950">
        <Link to="/" className="[&.active]:font-semibold text-white">
          Home
        </Link>
      </div>

      <Outlet />
      <TanStackRouterDevtools />
    </>
  ),
});
